import json
from pathlib import Path
from typing import Optional, Type

import numpy
import torch
import yaml
from acoustic_feature_extractor.data.phoneme import (
    BasePhoneme,
    JvsPhoneme,
    phoneme_type_to_class,
)
from acoustic_feature_extractor.data.sampling_data import SamplingData
from hifi_gan.env import AttrDict
from hifi_gan.models import Generator as HifiGanPredictor
from yukarin_s.config import Config as ConfigS
from yukarin_s.generator import Generator as GeneratorS
from yukarin_sa.config import Config as ConfigSa
from yukarin_sa.dataset import split_mora, unvoiced_mora_phoneme_list
from yukarin_sa.generator import Generator as GeneratorSa
from yukarin_soso.config import Config as ConfigSoso
from yukarin_soso.generator import Generator as YukarinSosoGenerator
from yukarin_sosoa.config import Config as ConfigSosoa
from yukarin_sosoa.generator import Generator as YukarinSosoaGenerator

from yukarin_soso_connector.full_context_label import extract_full_context_label
from yukarin_soso_connector.utility import get_predictor_model_path, remove_weight_norm


class Forwarder:
    def __init__(
        self,
        yukarin_s_model_dir: Path,
        yukarin_sa_model_dir: Path,
        yukarin_soso_model_dir: Optional[Path],
        yukarin_sosoa_model_dir: Optional[Path],
        hifigan_model_dir: Path,
        use_gpu: bool,
    ):
        super().__init__()

        # yukarin_s
        self.yukarin_s_phoneme_class: Optional[Type[BasePhoneme]] = None

        with yukarin_s_model_dir.joinpath("config.yaml").open() as f:
            config = ConfigS.from_dict(yaml.safe_load(f))

        yukarin_s_generator = GeneratorS(
            config=config,
            predictor=get_predictor_model_path(yukarin_s_model_dir),
            use_gpu=use_gpu,
        )
        yukarin_s_generator.predictor.apply(remove_weight_norm)

        self.yukarin_s_phoneme_class = phoneme_type_to_class[
            config.dataset.phoneme_type
        ]

        self.yukarin_s_generator = yukarin_s_generator
        print("yukarin_s loaded!")

        # yukarin_sa
        with yukarin_sa_model_dir.joinpath("config.yaml").open() as f:
            config = ConfigSa.from_dict(yaml.safe_load(f))

        yukarin_sa_generator = GeneratorSa(
            config=config,
            predictor=get_predictor_model_path(yukarin_sa_model_dir),
            use_gpu=use_gpu,
        )
        yukarin_sa_generator.predictor.apply(remove_weight_norm)

        assert (
            self.yukarin_s_phoneme_class
            is phoneme_type_to_class[config.dataset.phoneme_type]
        )

        assert yukarin_sa_generator.config.dataset.f0_process_mode == "voiced_mora"
        self.yukarin_sa_generator = yukarin_sa_generator
        print("yukarin_sa loaded!")

        # yukarin_soso or yukarin_sosoa
        self.yukarin_soso_phoneme_class: Optional[Type[BasePhoneme]] = None

        if yukarin_soso_model_dir is not None:
            with yukarin_soso_model_dir.joinpath("config.yaml").open() as f:
                config = ConfigSoso.from_dict(yaml.safe_load(f))

            yukarin_soso_generator = YukarinSosoGenerator(
                config=config,
                predictor=get_predictor_model_path(yukarin_soso_model_dir),
                use_gpu=use_gpu,
            )
            yukarin_soso_generator.predictor.apply(remove_weight_norm)

            self.yukarin_soso_phoneme_class = phoneme_type_to_class[
                config.dataset.phoneme_type
            ]

            self.yukarin_soso_generator = yukarin_soso_generator
            print("yukarin_soso loaded!")

        else:
            self.yukarin_soso_generator = None

        if yukarin_sosoa_model_dir is not None:
            with yukarin_sosoa_model_dir.joinpath("config.yaml").open() as f:
                config = ConfigSosoa.from_dict(yaml.safe_load(f))

            yukarin_sosoa_generator = YukarinSosoaGenerator(
                config=config,
                predictor=get_predictor_model_path(yukarin_sosoa_model_dir),
                use_gpu=use_gpu,
            )
            yukarin_sosoa_generator.predictor.apply(remove_weight_norm)

            self.yukarin_soso_phoneme_class = phoneme_type_to_class[
                config.dataset.phoneme_type
            ]

            self.yukarin_sosoa_generator = yukarin_sosoa_generator
            print("yukarin_sosoa loaded!")

        else:
            self.yukarin_sosoa_generator = None

        # hifi-gan
        device = yukarin_s_generator.device
        vocoder_model_config = AttrDict(
            json.loads((hifigan_model_dir / "config.json").read_text())
        )

        hifi_gan_predictor = HifiGanPredictor(vocoder_model_config).to(device)
        checkpoint_dict = torch.load(
            get_predictor_model_path(hifigan_model_dir, prefix="g_", postfix=""),
            map_location=device,
        )
        hifi_gan_predictor.load_state_dict(checkpoint_dict["generator"])
        hifi_gan_predictor.eval()
        hifi_gan_predictor.remove_weight_norm()

        self.hifi_gan_predictor = hifi_gan_predictor
        print("hifi-gan loaded!")

        self.device = device

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

        phoneme_length = self.yukarin_s_generator.generate(
            phoneme_list=phoneme_list_s, speaker_id=f0_speaker_id
        )
        phoneme_length[0] = phoneme_length[-1] = 0.1
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

        f0_list = self.yukarin_sa_generator.generate(
            vowel_phoneme_list=vowel_phoneme_list[numpy.newaxis],
            consonant_phoneme_list=consonant_phoneme_list[numpy.newaxis],
            start_accent_list=start_accent_list[vowel_indexes][numpy.newaxis],
            end_accent_list=end_accent_list[vowel_indexes][numpy.newaxis],
            start_accent_phrase_list=start_accent_phrase_list[vowel_indexes][
                numpy.newaxis
            ],
            end_accent_phrase_list=end_accent_phrase_list[vowel_indexes][numpy.newaxis],
            speaker_id=f0_speaker_id,
        )[0]
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

        # forward yukarin soso
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

        if self.yukarin_soso_generator is not None:
            spec = self.yukarin_soso_generator.generate(
                f0=[f0[:, numpy.newaxis]],
                phoneme=[phoneme],
                speaker_id=numpy.array(speaker_id).reshape(-1),
            )[0]
        else:
            spec = self.yukarin_sosoa_generator.generate(
                f0_list=[f0[:, numpy.newaxis]],
                phoneme_list=[phoneme],
                speaker_id=numpy.array(speaker_id).reshape(-1),
            )[0]

        # forward hifi gan
        x = spec.T
        wave = self.hifi_gan_predictor(
            torch.FloatTensor(x).unsqueeze(0).to(self.device)
        ).squeeze()

        return wave.cpu().numpy()
