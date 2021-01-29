import argparse
import json
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
    phoneme_list = openjtalk_label_getter(
        text=text,
        openjtalk_command="open_jtalk",
        dict_path=Path("/var/lib/mecab/dic/open-jtalk/naist-jdic"),
        htsvoice_path=Path(
            "/usr/share/hts-voice/nitech-jp-atr503-m001/nitech_jp_atr503_m001.htsvoice"
        ),
        output_wave_path=None,
        output_log_path=None,
        output_type=OutputType.phoneme,
    )
    print(phoneme_list)
    json.dump([p.label for p in phoneme_list], open("hiho_phoneme_list.json", mode="w"))

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
    print(phoneme_length)
    numpy.save("hiho_phoneme_length.npy", phoneme_length)

    # yukarin_sos
    with open("data/yukarin_sos/config.yaml") as f:
        d = yaml.safe_load(f)

    generator_sos = GeneratorSos(
        config=ConfigSos.from_dict(d),
        predictor=Path("data/yukarin_sos/predictor_10000.pth"),
        use_gpu=False,
    )

    rate = 200

    phoneme_list = numpy.r_[0, phoneme_list, 0]
    phoneme_length = numpy.r_[0.1, phoneme_length, 0.1]
    phoneme = numpy.repeat(
        phoneme_list, numpy.round(phoneme_length * rate).astype(numpy.int32)
    )

    f0 = generator_sos.generate(
        phoneme=phoneme[numpy.newaxis],
        speaker_id=numpy.array(speaker_id).reshape(1),
    )[0]
    print(f0)
    numpy.save("hiho_f0.npy", f0)

    # yukarin_soso
    with open("data/yukarin_soso/config.yaml") as f:
        d = yaml.safe_load(f)

    generator_soso = GeneratorSoso(
        config=ConfigSoso.from_dict(d),
        predictor=Path("data/yukarin_soso/predictor_820000.pth"),
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
    print(spec)
    numpy.save("hiho_spec.npy", spec)

    # hifi-gan
    wave = inference_hifigan(
        x=spec.T,
        checkpoint_file="data/hifigan/g_01460000",
        config_file="data/hifigan/config.json",
    )

    # save
    soundfile.write("hiho_output.wav", data=wave, samplerate=24000)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("text")
    parser.add_argument("speaker_id", type=int)
    run(**vars(parser.parse_args()))
