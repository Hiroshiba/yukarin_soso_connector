import argparse
import json
import re
from pathlib import Path

import numpy
import soundfile
import yaml
from acoustic_feature_extractor.data.phoneme import JvsPhoneme
from acoustic_feature_extractor.data.sampling_data import SamplingData
from openjtalk_label_getter import OutputType, openjtalk_label_getter
from yukarin_s.config import Config as ConfigS
from yukarin_s.generator import Generator as GeneratorS
from yukarin_sos.config import Config as ConfigSos
from yukarin_sos.generator import Generator as GeneratorSos
from yukarin_soso.config import Config as ConfigSoso
from yukarin_soso.generator import Generator as GeneratorSoso

from inference_hifigan import inference_hifigan


def run(text: str, speaker_id: int):
    # phoneme
    label_list = openjtalk_label_getter(
        text=text,
        openjtalk_command="open_jtalk",
        dict_path=Path("/var/lib/mecab/dic/open-jtalk/naist-jdic"),
        htsvoice_path=Path(
            "/usr/share/hts-voice/nitech-jp-atr503-m001/nitech_jp_atr503_m001.htsvoice"
        ),
        output_wave_path="hiho_openjtalk_wave.wav",
        output_log_path="hiho_openjtalk_log.txt",
        output_type=OutputType.full_context_label,
    )
    json.dump([p.label for p in label_list], open("hiho_phoneme_list.json", mode="w"))

    is_type1 = False
    phoneme_list = label_list
    start_accent_list = []
    end_accent_list = []
    for i, label in enumerate(label_list):
        phoneme, a, b = re.match(
            r".+?\^.+?\-(.+?)\+.+?\=.+?/A:(.+?)\+(.+?)\+.+?/B:.+", label.label
        ).groups()

        is_end_accent = a == "0"

        if b == "1":
            is_type1 = is_end_accent

        if b == "1" and is_type1:
            is_start_accent = True
        elif b == "2" and not is_type1:
            is_start_accent = True
        else:
            is_start_accent = False

        phoneme_list[i].label = phoneme
        start_accent_list.append(is_start_accent)
        end_accent_list.append(is_end_accent)

    # yukarin_s
    with open("data/yukarin_s/config.yaml") as f:
        d = yaml.safe_load(f)

    generator_s = GeneratorS(
        config=ConfigS.from_dict(d),
        predictor=Path("data/yukarin_s/predictor_20000.pth"),
        use_gpu=False,
    )

    phoneme_list = [
        JvsPhoneme(phoneme=p.label, start=p.start, end=p.end)
        for p in phoneme_list[1:-1]
    ]
    phoneme_list = numpy.array([p.phoneme_id for p in phoneme_list])

    phoneme_length = generator_s.generate(
        phoneme_list=phoneme_list, speaker_id=speaker_id
    )
    numpy.save("hiho_phoneme_length.npy", phoneme_length)

    # yukarin_sos
    with open("data/yukarin_sos/config.yaml") as f:
        d = yaml.safe_load(f)

    generator_sos = GeneratorSos(
        config=ConfigSos.from_dict(d),
        predictor=Path("data/yukarin_sos/predictor_70000.pth"),
        use_gpu=False,
    )

    rate = 200

    phoneme_list = numpy.r_[0, phoneme_list, 0]
    start_accent_list = numpy.array(start_accent_list, dtype=numpy.int64)
    end_accent_list = numpy.array(end_accent_list, dtype=numpy.int64)
    phoneme_length = numpy.r_[0.1, phoneme_length, 0.1]
    phoneme = numpy.repeat(
        phoneme_list, numpy.round(phoneme_length * rate).astype(numpy.int32)
    )
    start_accent = numpy.repeat(
        start_accent_list, numpy.round(phoneme_length * rate).astype(numpy.int32)
    )
    end_accent = numpy.repeat(
        end_accent_list, numpy.round(phoneme_length * rate).astype(numpy.int32)
    )

    f0 = generator_sos.generate(
        phoneme=phoneme[numpy.newaxis],
        start_accent=start_accent[numpy.newaxis],
        end_accent=end_accent[numpy.newaxis],
        speaker_id=numpy.array(speaker_id).reshape(1),
    )[0]
    numpy.save("hiho_f0.npy", f0)

    # yukarin_soso
    with open("data/yukarin_soso/config.yaml") as f:
        d = yaml.safe_load(f)

    generator_soso = GeneratorSoso(
        config=ConfigSoso.from_dict(d),
        predictor=Path("data/yukarin_soso/predictor_120000.pth"),
        use_gpu=False,
    )

    array = numpy.zeros((len(phoneme), JvsPhoneme.num_phoneme), dtype=numpy.float32)
    array[numpy.arange(len(phoneme)), phoneme] = 1
    phoneme = array

    f0 = SamplingData(array=f0, rate=200).resample(24000 / 256)
    phoneme = SamplingData(array=phoneme, rate=200).resample(24000 / 256)

    spec = generator_soso.generate(
        f0=f0[numpy.newaxis, :, numpy.newaxis],
        phoneme=phoneme[numpy.newaxis],
        speaker_id=numpy.array(speaker_id).reshape(1),
    )[0]
    numpy.save("hiho_spec.npy", spec)

    # hifi-gan
    wave = inference_hifigan(
        x=spec.T,
        checkpoint_file="data/hifigan/g_01460000",
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
