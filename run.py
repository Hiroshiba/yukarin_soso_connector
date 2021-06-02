import argparse
import json
import re
from pathlib import Path
from typing import List

import numpy
import pyopenjtalk
import soundfile
import yaml
from acoustic_feature_extractor.data.phoneme import JvsPhoneme
from acoustic_feature_extractor.data.sampling_data import SamplingData
from yukarin_s.config import Config as ConfigS
from yukarin_s.generator import Generator as GeneratorS
from yukarin_sa.config import Config as ConfigSa
from yukarin_sa.dataset import split_mora, unvoiced_mora_phoneme_list
from yukarin_sa.generator import Generator as GeneratorSa
from yukarin_soso.config import Config as ConfigSoso
from yukarin_soso.generator import Generator as GeneratorSoso

from inference_hifigan import inference_hifigan
from yukarin_soso_connector.full_context_label import extract_full_context_label


def f0_mean(f0: numpy.ndarray, rate: float, split_second_list: List[float]):
    indexes = numpy.floor(numpy.array(split_second_list) * rate).astype(int)
    for a in numpy.split(f0, indexes):
        a[:] = numpy.mean(a[a > 0])
    f0[numpy.isnan(f0)] = 0
    return f0


def _extract_number(f):
    s = re.findall(r"\d+", str(f))
    return int(s[-1]) if s else -1


def _get_predictor_model_path(
    model_dir: Path,
    iteration: int = None,
    prefix: str = "predictor_",
):
    if iteration is None:
        paths = model_dir.glob(prefix + "*.pth")
        model_path = list(sorted(paths, key=_extract_number))[-1]
    else:
        model_path = model_dir / (prefix + "{}.pth".format(iteration))
        assert model_path.exists()
    return model_path


def run(text: str, speaker_id: int):
    rate = 200

    # phoneme
    utterance = extract_full_context_label(text)

    # utterance.breath_groups[0].accent_phrases[2].accent = 2
    # utterance.breath_groups[1].accent_phrases[1].accent = 6
    # utterance.breath_groups[1].accent_phrases[3].accent = 5

    x, sr = pyopenjtalk.synthesize(utterance.labels, speed=1, half_tone=0)
    x /= 2 ** 16
    soundfile.write("hiho_openjtalk_wave.wav", x, sr)

    label_data_list = utterance.phonemes

    json.dump(
        [p.label for p in label_data_list], open("hiho_label_list.json", mode="w")
    )

    is_type1 = False
    phoneme_str_list = []
    start_accent_list = numpy.ones(len(label_data_list), dtype=numpy.int64) * numpy.nan
    end_accent_list = numpy.ones(len(label_data_list), dtype=numpy.int64) * numpy.nan
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
    start_accent_phrase_list = numpy.array(start_accent_phrase_list, dtype=numpy.int64)
    end_accent_phrase_list = numpy.array(end_accent_phrase_list, dtype=numpy.int64)

    json.dump(phoneme_str_list, open("hiho_phoneme_list.json", mode="w"))

    # yukarin_s
    with open("data/yukarin_s/check-bs128-hs32/config.yaml") as f:
        d = yaml.safe_load(f)

    generator_s = GeneratorS(
        config=ConfigS.from_dict(d),
        predictor=Path("data/yukarin_s/check-bs128-hs32/predictor_50000.pth"),
        use_gpu=False,
    )

    phoneme_data_list = [
        JvsPhoneme(phoneme=p, start=i, end=i + 1)
        for i, p in enumerate(phoneme_str_list)
    ]
    phoneme_data_list = JvsPhoneme.convert(phoneme_data_list)
    phoneme_list_s = numpy.array([p.phoneme_id for p in phoneme_data_list])

    phoneme_length = generator_s.generate(
        phoneme_list=phoneme_list_s,
        speaker_id=speaker_id,
    )
    phoneme_length[0] = phoneme_length[-1] = 0.1
    phoneme_length = numpy.round(phoneme_length * rate) / rate
    numpy.save("hiho_phoneme_length.npy", phoneme_length)

    # yukarin_sa
    model_dir = Path(
        "data/yukarin_sa/withjsss-lr1.0e-03-ehs32-aehs32-pl2-pn8-fl2-fn2-try1"
    )
    with (model_dir / "config.yaml").open() as f:
        d = yaml.safe_load(f)

    generator_sa = GeneratorSa(
        config=ConfigSa.from_dict(d),
        predictor=_get_predictor_model_path(model_dir),
        use_gpu=False,
    )

    assert generator_sa.config.dataset.f0_process_mode == "voiced_mora"
    (
        consonant_phoneme_data_list,
        vowel_phoneme_data_list,
        vowel_indexes_data,
    ) = split_mora(phoneme_data_list)

    vowel_indexes = numpy.array(vowel_indexes_data)

    vowel_phoneme_list = numpy.array([p.phoneme_id for p in vowel_phoneme_data_list])
    consonant_phoneme_list = numpy.array(
        [p.phoneme_id if p is not None else -1 for p in consonant_phoneme_data_list]
    )
    phoneme_length_sa = numpy.array(
        [a.sum() for a in numpy.split(phoneme_length, vowel_indexes[:-1] + 1)]
    )

    f0_list = generator_sa.generate(
        vowel_phoneme_list=vowel_phoneme_list[numpy.newaxis],
        consonant_phoneme_list=consonant_phoneme_list[numpy.newaxis],
        start_accent_list=start_accent_list[vowel_indexes][numpy.newaxis],
        end_accent_list=end_accent_list[vowel_indexes][numpy.newaxis],
        start_accent_phrase_list=start_accent_phrase_list[vowel_indexes][numpy.newaxis],
        end_accent_phrase_list=end_accent_phrase_list[vowel_indexes][numpy.newaxis],
        speaker_id=speaker_id,
    )[0]

    for i, p in enumerate(vowel_phoneme_data_list):
        if p.phoneme in unvoiced_mora_phoneme_list:
            f0_list[i] = 0

    numpy.save("hiho_f0_list.npy", f0_list)

    phoneme = numpy.repeat(
        phoneme_list_s, numpy.round(phoneme_length * rate).astype(numpy.int32)
    )
    f0 = numpy.repeat(
        f0_list, numpy.round(phoneme_length_sa * rate).astype(numpy.int32)
    )

    numpy.save("hiho_f0.npy", f0)

    # yukarin_soso
    with open(
        "data/yukarin_soso/f0mean-wei_voicedmora-sl1280-bs128-lr1.0e-03-mt0.2-mn32-try1/config.yaml"
    ) as f:
        d = yaml.safe_load(f)

    generator_soso = GeneratorSoso(
        config=ConfigSoso.from_dict(d),
        predictor=Path(
            "data/yukarin_soso/f0mean-wei_voicedmora-sl1280-bs128-lr1.0e-03-mt0.2-mn32-try1/predictor_220000.pth"
        ),
        use_gpu=False,
    )
    assert generator_soso.config.dataset.f0_process_mode == "voiced_mora_mean"

    array = numpy.zeros((len(phoneme), JvsPhoneme.num_phoneme), dtype=numpy.float32)
    array[numpy.arange(len(phoneme)), phoneme] = 1
    phoneme = array

    f0 = SamplingData(array=f0, rate=rate).resample(24000 / 256)
    phoneme = SamplingData(array=phoneme, rate=rate).resample(24000 / 256)

    spec = generator_soso.generate(
        f0=f0[numpy.newaxis, :, numpy.newaxis],
        phoneme=phoneme[numpy.newaxis],
        speaker_id=numpy.array(speaker_id).reshape(-1),
    )[0]
    numpy.save("hiho_spec.npy", spec)

    # hifi-gan
    wave = inference_hifigan(
        x=spec.T,
        checkpoint_file="data/hifigan/g_03080000",
        config_file="data/hifigan/config.json",
    )

    # save
    soundfile.write("hiho_output.wav", data=wave, samplerate=24000)
    soundfile.write(f"{text}-{speaker_id}.wav", data=wave, samplerate=24000)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("text")
    parser.add_argument("speaker_id", type=int)
    run(**vars(parser.parse_args()))
