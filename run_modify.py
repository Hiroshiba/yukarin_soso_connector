import argparse

import pyopenjtalk
import soundfile

from yukarin_soso_connector.full_context_label import extract_full_context_label


def run(text: str):
    utt = extract_full_context_label(text)

    x, sr = pyopenjtalk.synthesize(utt.labels, speed=1, half_tone=0)
    x /= 2 ** 16
    soundfile.write("hiho_output_before.wav", x, sr)

    utt.breath_groups[0].accent_phrases[2].accent = 2
    utt.breath_groups[1].accent_phrases[1].accent = 6
    utt.breath_groups[1].accent_phrases[3].accent = 5

    x, sr = pyopenjtalk.synthesize(utt.labels, speed=1, half_tone=0)
    x /= 2 ** 16
    soundfile.write("hiho_output_after.wav", x, sr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("text")
    run(**vars(parser.parse_args()))
