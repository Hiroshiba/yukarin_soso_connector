from typing import Any, List, Optional, Union

import numpy
import torch
from acoustic_feature_extractor.data.phoneme import JvsPhoneme, OjtPhoneme
from acoustic_feature_extractor.data.sampling_data import SamplingData
from torch import Tensor, nn
from yukarin_sa.dataset import split_mora, unvoiced_mora_phoneme_list
from yukarin_soso_connector.forwarder import Forwarder
from yukarin_soso_connector.full_context_label import extract_full_context_label
from yukarin_soso_connector.jit_forwarder.jit_yukarin_s import JitYukarinS
from yukarin_soso_connector.jit_forwarder.jit_yukarin_sa import JitYukarinSa
from yukarin_soso_connector.jit_forwarder.jit_yukarin_sosoa import JitYukarinSosoa


class JitDecodeForwarder(nn.Module):
    def __init__(
        self,
        yukarin_sosoa_forwarder: nn.Module,
        hifi_gan_forwarder: nn.Module,
    ):
        super().__init__()
        self.yukarin_sosoa_forwarder = yukarin_sosoa_forwarder
        self.hifi_gan_forwarder = hifi_gan_forwarder

    def forward(
        self,
        f0_list: List[Tensor],
        phoneme_list: List[Tensor],
        speaker_id: Optional[Tensor] = None,
    ):
        # forward sosoa
        spec = self.yukarin_sosoa_forwarder(
            f0_list=f0_list, phoneme_list=phoneme_list, speaker_id=speaker_id
        )[0]

        # forward hifi gan
        x = spec.T
        wave = self.hifi_gan_forwarder(x.unsqueeze(0)).squeeze()
        return wave


class JitForwarder(nn.Module):
    def __init__(
        self,
        forwarder: Forwarder = None,
        yukarin_s_forwarder=None,
        yukarin_sa_forwarder=None,
        decode_forwarder=None,
        device=None,
    ):
        super().__init__()

        if forwarder is not None:
            self.device = forwarder.device

            # yukarin_s
            self.yukarin_s_phoneme_class = forwarder.yukarin_s_phoneme_class
            self.yukarin_s_forwarder = torch.jit.script(
                JitYukarinS(forwarder.yukarin_s_generator.predictor)
            )
            print("--- yukarin_s forwarder ---\n", self.yukarin_s_forwarder.code)

            # yukarin_sa
            self.yukarin_sa_forwarder = torch.jit.script(
                JitYukarinSa(forwarder.yukarin_sa_generator.predictor)
            )
            print("--- yukarin_sa forwarder ---\n", self.yukarin_sa_forwarder.code)

            # yukarin_soso or yukarin_sosoa
            self.yukarin_soso_phoneme_class = forwarder.yukarin_soso_phoneme_class
            self.yukarin_soso_forwarder = None
            assert forwarder.yukarin_soso_generator is None

            yukarin_sosoa_forwarder = torch.jit.script(
                JitYukarinSosoa(
                    forwarder.yukarin_sosoa_generator.predictor, device=self.device
                )
            )
            print("--- yukarin_sosoa forwarder ---\n", yukarin_sosoa_forwarder.code)

            # hifi-gan
            hifi_gan_forwarder = torch.jit.trace(
                forwarder.hifi_gan_predictor,
                torch.rand(
                    1,
                    forwarder.hifi_gan_predictor.conv_pre.in_channels,
                    1,
                    device=self.device,
                ),
            )
            print("--- hifi_gan forwarder ---\n", hifi_gan_forwarder.code)

            # decode
            self.decode_forwarder = torch.jit.script(
                JitDecodeForwarder(
                    yukarin_sosoa_forwarder=yukarin_sosoa_forwarder,
                    hifi_gan_forwarder=hifi_gan_forwarder,
                )
            )

        else:
            self.yukarin_s_forwarder = yukarin_s_forwarder
            self.yukarin_sa_forwarder = yukarin_sa_forwarder
            self.yukarin_soso_forwarder = None
            self.decode_forwarder = decode_forwarder
            self.yukarin_s_phoneme_class = OjtPhoneme
            self.yukarin_soso_phoneme_class = OjtPhoneme
            self.device = device

    def to_tensor(self, array: Union[Tensor, numpy.ndarray, Any]):
        if not isinstance(array, (Tensor, numpy.ndarray)):
            array = numpy.asarray(array)
        if isinstance(array, numpy.ndarray):
            array = torch.from_numpy(array)
        return array.to(self.device)

    @torch.no_grad()
    def forward(
        self, text: str, speaker_id: int, f0_speaker_id: int, f0_correct: float = 0
    ):
        rate = 200

        # phoneme
        utterance = extract_full_context_label(text)
        label_data_list = utterance.phonemes

        is_type1 = False
        phoneme_str_list = []
        start_accent_list = (
            numpy.ones(len(label_data_list), dtype=numpy.int64) * numpy.nan
        )
        end_accent_list = (
            numpy.ones(len(label_data_list), dtype=numpy.int64) * numpy.nan
        )
        start_accent_phrase_list = (
            numpy.ones(len(label_data_list), dtype=numpy.int64) * numpy.nan
        )
        end_accent_phrase_list = (
            numpy.ones(len(label_data_list), dtype=numpy.int64) * numpy.nan
        )
        for i, label in enumerate(label_data_list):
            is_end_accent = label.contexts["a1"] == "0"

            if label.contexts["a2"] == "1":
                is_type1 = is_end_accent

            if label.contexts["a2"] == "1" and is_type1:
                is_start_accent = True
            elif label.contexts["a2"] == "2" and not is_type1:
                is_start_accent = True
            else:
                is_start_accent = False

            phoneme_str_list.append(label.phoneme)
            start_accent_list[i] = is_start_accent
            end_accent_list[i] = is_end_accent
            start_accent_phrase_list[i] = label.contexts["a2"] == "1"
            end_accent_phrase_list[i] = label.contexts["a3"] == "1"

        start_accent_list = numpy.array(start_accent_list, dtype=numpy.int64)
        end_accent_list = numpy.array(end_accent_list, dtype=numpy.int64)
        start_accent_phrase_list = numpy.array(
            start_accent_phrase_list, dtype=numpy.int64
        )
        end_accent_phrase_list = numpy.array(end_accent_phrase_list, dtype=numpy.int64)

        # forward yukarin s
        assert self.yukarin_s_phoneme_class is not None

        phoneme_data_list = [
            self.yukarin_s_phoneme_class(phoneme=p, start=i, end=i + 1)
            for i, p in enumerate(phoneme_str_list)
        ]
        phoneme_data_list = self.yukarin_s_phoneme_class.convert(phoneme_data_list)
        phoneme_list_s = numpy.array([p.phoneme_id for p in phoneme_data_list])

        phoneme_length = (
            self.yukarin_s_forwarder(
                phoneme_list=self.to_tensor(phoneme_list_s),
                speaker_id=self.to_tensor(f0_speaker_id),
            )
            .cpu()
            .numpy()
        )
        phoneme_length[0] = phoneme_length[-1] = 0.5
        phoneme_length = numpy.round(phoneme_length * rate) / rate

        # forward yukarin sa
        (
            consonant_phoneme_data_list,
            vowel_phoneme_data_list,
            vowel_indexes_data,
        ) = split_mora(phoneme_data_list)

        vowel_indexes = numpy.array(vowel_indexes_data)

        vowel_phoneme_list = numpy.array(
            [p.phoneme_id for p in vowel_phoneme_data_list]
        )
        consonant_phoneme_list = numpy.array(
            [p.phoneme_id if p is not None else -1 for p in consonant_phoneme_data_list]
        )
        phoneme_length_sa = numpy.array(
            [a.sum() for a in numpy.split(phoneme_length, vowel_indexes[:-1] + 1)]
        )

        f0_list = (
            self.yukarin_sa_forwarder(
                vowel_phoneme_list=self.to_tensor(vowel_phoneme_list[numpy.newaxis]),
                consonant_phoneme_list=self.to_tensor(
                    consonant_phoneme_list[numpy.newaxis]
                ),
                start_accent_list=self.to_tensor(
                    start_accent_list[vowel_indexes][numpy.newaxis]
                ),
                end_accent_list=self.to_tensor(
                    end_accent_list[vowel_indexes][numpy.newaxis]
                ),
                start_accent_phrase_list=self.to_tensor(
                    start_accent_phrase_list[vowel_indexes][numpy.newaxis]
                ),
                end_accent_phrase_list=self.to_tensor(
                    end_accent_phrase_list[vowel_indexes][numpy.newaxis]
                ),
                speaker_id=self.to_tensor(f0_speaker_id),
            )
            .cpu()
            .numpy()[0]
        )
        f0_list += f0_correct

        for i, p in enumerate(vowel_phoneme_data_list):
            if p.phoneme in unvoiced_mora_phoneme_list:
                f0_list[i] = 0

        phoneme = numpy.repeat(
            phoneme_list_s, numpy.round(phoneme_length * rate).astype(numpy.int32)
        )
        f0 = numpy.repeat(
            f0_list, numpy.round(phoneme_length_sa * rate).astype(numpy.int32)
        )

        # forward decode
        assert self.yukarin_soso_phoneme_class is not None

        if (
            self.yukarin_soso_phoneme_class is not JvsPhoneme
            and self.yukarin_soso_phoneme_class is not self.yukarin_s_phoneme_class
        ):
            phoneme = numpy.array(
                [
                    self.yukarin_soso_phoneme_class.phoneme_list.index(
                        JvsPhoneme.phoneme_list[p]
                    )
                    for p in phoneme
                ],
                dtype=numpy.int32,
            )

        array = numpy.zeros(
            (len(phoneme), self.yukarin_soso_phoneme_class.num_phoneme),
            dtype=numpy.float32,
        )
        array[numpy.arange(len(phoneme)), phoneme] = 1
        phoneme = array

        f0 = SamplingData(array=f0, rate=rate).resample(24000 / 256)
        phoneme = SamplingData(array=phoneme, rate=rate).resample(24000 / 256)

        wave = self.decode_forwarder(
            f0_list=[self.to_tensor(f0[:, numpy.newaxis])],
            phoneme_list=[self.to_tensor(phoneme)],
            speaker_id=self.to_tensor(numpy.array(speaker_id).reshape(-1)),
        )
        return wave.cpu().numpy()
