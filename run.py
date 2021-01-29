import argparse
from pathlib import Path

from openjtalk_label_getter import OutputType, openjtalk_label_getter


def run(text: str):
    phoneme = [
        label.label
        for label in openjtalk_label_getter(
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
    ]
    print(phoneme)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("text")
    run(**vars(parser.parse_args()))
