import argparse
from itertools import product
from pathlib import Path
from typing import List, Optional

import each_cpp_forwarder
import soundfile
from tqdm import tqdm

from yukarin_soso_connector.cpp_forwarder import CppForwarder


def run(
    forwarder_dir: Path,
    use_gpu: bool,
    texts: List[str],
    speaker_ids: List[int],
    f0_speaker_ids: Optional[List[int]],
    f0_correct: float,
):
    if f0_speaker_ids is None or len(f0_speaker_ids) == 0:
        f0_speaker_ids = speaker_ids
    else:
        assert len(speaker_ids) == len(f0_speaker_ids)

    device = "cuda" if use_gpu else "cpu"
    each_cpp_forwarder.initialize(
        str(forwarder_dir.joinpath(f"hiho_yukarin_s_script_{device}.pt")),
        str(forwarder_dir.joinpath(f"hiho_yukarin_sa_script_{device}.pt")),
        str(forwarder_dir.joinpath(f"hiho_decode_script_{device}.pt")),
        use_gpu,
    )

    forwarder = CppForwarder(
        yukarin_s_forwarder=each_cpp_forwarder.yukarin_s_forward,
        yukarin_sa_forwarder=each_cpp_forwarder.yukarin_sa_forward,
        decode_forwarder=each_cpp_forwarder.decode_forward,
    )

    for text, (speaker_id, f0_speaker_id) in tqdm(
        list(product(texts, zip(speaker_ids, f0_speaker_ids)))
    ):
        wave = forwarder.forward(
            text=text,
            speaker_id=speaker_id,
            f0_speaker_id=f0_speaker_id,
            f0_correct=f0_correct,
        )

        soundfile.write(f"{text}-{speaker_id}.wav", data=wave, samplerate=24000)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--forwarder_dir", type=Path, required=True)
    parser.add_argument("--use_gpu", action="store_true")
    parser.add_argument("--texts", nargs="+", required=True)
    parser.add_argument("--speaker_ids", nargs="+", type=int, required=True)
    parser.add_argument("--f0_speaker_ids", nargs="*", type=int)
    parser.add_argument("--f0_correct", type=float, default=0)
    run(**vars(parser.parse_args()))
