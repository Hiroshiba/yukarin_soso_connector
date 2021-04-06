import argparse
import json
import re
from pathlib import Path
from pprint import pprint
from typing import List

import numpy
import soundfile
import yaml
from acoustic_feature_extractor.data.f0 import F0, F0Type
from acoustic_feature_extractor.data.phoneme import JvsPhoneme
from acoustic_feature_extractor.data.sampling_data import SamplingData
from acoustic_feature_extractor.data.wave import Wave
from openjtalk_label_getter import OutputType, openjtalk_label_getter
from yukarin_soso.config import Config as ConfigSoso
from yukarin_soso.generator import Generator as GeneratorSoso

from inference_hifigan import inference_hifigan

mora_phoneme_list = ["a", "i", "u", "e", "o", "A", "I", "U", "E", "O", "N", "cl", "pau"]


def run(text: str, speaker_id: int):
    rate = 200

    # phoneme
    openjtalk_wave_path = Path("hiho_openjtalk_wave.wav")
    phoneme_list = openjtalk_label_getter(
        text=text,
        openjtalk_command="open_jtalk",
        dict_path=Path("/var/lib/mecab/dic/open-jtalk/naist-jdic"),
        htsvoice_path=Path("tohoku-f01-neutral.htsvoice"),
        output_wave_path=openjtalk_wave_path,
        output_log_path="hiho_openjtalk_log.txt",
        output_type=OutputType.phoneme,
    )
    json.dump([p.label for p in phoneme_list], open("hiho_phoneme_list.json", mode="w"))

    # analyze
    wave = Wave.load(openjtalk_wave_path)
    f0 = F0.from_wave(
        wave,
        frame_period=5,
        f0_floor=70,
        f0_ceil=600,
        with_vuv=False,
        f0_type=F0Type.true_world,
    ).array.astype(numpy.float32)

    phoneme_list = JvsPhoneme.convert(
        [JvsPhoneme(phoneme=p.label, start=p.start, end=p.end) for p in phoneme_list]
    )
    phoneme_length = numpy.array([p.end - p.start for p in phoneme_list])
    phoneme = numpy.array([p.phoneme_id for p in phoneme_list])
    phoneme = numpy.repeat(
        phoneme,
        numpy.round(phoneme_length * rate).astype(numpy.int32),
    )

    min_length = min(len(f0), len(phoneme))
    f0 = f0[:min_length]
    phoneme = phoneme[:min_length]

    # yukarin_soso
    with open("data/yukarin_soso/sota/config.yaml") as f:
        d = yaml.safe_load(f)

    generator_soso = GeneratorSoso(
        config=ConfigSoso.from_dict(d),
        predictor=Path("data/yukarin_soso/sota/predictor_980000.pth"),
        use_gpu=False,
    )

    array = numpy.zeros((len(phoneme), JvsPhoneme.num_phoneme), dtype=numpy.float32)
    array[numpy.arange(len(phoneme)), phoneme] = 1
    phoneme = array

    f0 = SamplingData(array=f0, rate=rate).resample(24000 / 256)
    phoneme = SamplingData(array=phoneme, rate=rate).resample(24000 / 256)

    spec = generator_soso.generate(
        f0=f0[numpy.newaxis, :, numpy.newaxis],
        phoneme=phoneme[numpy.newaxis],
        speaker_id=numpy.array(speaker_id).reshape(1),
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
