import argparse
from itertools import product
from pathlib import Path
from typing import List, Optional

import soundfile
import torch
from tqdm import tqdm

from yukarin_soso_connector.forwarder import Forwarder
from yukarin_soso_connector.jit_forwarder.jit_forwarder import JitForwarder


def run(
    yukarin_s_model_dir: Path,
    yukarin_sa_model_dir: Path,
    yukarin_soso_model_dir: Optional[Path],
    yukarin_sosoa_model_dir: Optional[Path],
    hifigan_model_dir: Path,
    hifigan_model_iteration: Optional[str],
    use_gpu: bool,
    texts: List[str],
    speaker_ids: List[int],
    f0_speaker_id: Optional[int],
    f0_correct: float,
    with_load: bool,
):
    if not with_load:
        forwarder = JitForwarder(
            Forwarder(
                yukarin_s_model_dir=yukarin_s_model_dir,
                yukarin_sa_model_dir=yukarin_sa_model_dir,
                yukarin_soso_model_dir=yukarin_soso_model_dir,
                yukarin_sosoa_model_dir=yukarin_sosoa_model_dir,
                hifigan_model_dir=hifigan_model_dir,
                hifigan_model_iteration=hifigan_model_iteration,
                use_gpu=use_gpu,
            )
        )

        device = "cuda" if use_gpu else "cpu"
        torch.jit.save(
            forwarder.yukarin_s_forwarder, f"hiho_yukarin_s_script_{device}.pt"
        )
        torch.jit.save(
            forwarder.yukarin_sa_forwarder, f"hiho_yukarin_sa_script_{device}.pt"
        )
        torch.jit.save(forwarder.decode_forwarder, f"hiho_decode_script_{device}.pt")

    else:
        device = "cuda" if use_gpu else "cpu"
        forwarder = JitForwarder(
            yukarin_s_forwarder=torch.jit.load(
                f"hiho_yukarin_s_script_{device}.pt", device
            ),
            yukarin_sa_forwarder=torch.jit.load(
                f"hiho_yukarin_sa_script_{device}.pt", device
            ),
            decode_forwarder=torch.jit.load(f"hiho_decode_script_{device}.pt", device),
            device=device,
        )

    for text, speaker_id in tqdm(list(product(texts, speaker_ids))):
        wave = forwarder.forward(
            text=text,
            speaker_id=speaker_id,
            f0_speaker_id=f0_speaker_id if f0_speaker_id is not None else speaker_id,
            f0_correct=f0_correct,
        )

        soundfile.write(f"{text}-{speaker_id}.wav", data=wave, samplerate=24000)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--yukarin_s_model_dir", type=Path, required=True)
    parser.add_argument("--yukarin_sa_model_dir", type=Path, required=True)
    parser.add_argument("--yukarin_soso_model_dir", type=Path)
    parser.add_argument("--yukarin_sosoa_model_dir", type=Path)
    parser.add_argument("--hifigan_model_dir", type=Path, required=True)
    parser.add_argument("--hifigan_model_iteration")
    parser.add_argument("--use_gpu", action="store_true")
    parser.add_argument("--texts", nargs="+", required=True)
    parser.add_argument("--speaker_ids", nargs="+", type=int, required=True)
    parser.add_argument("--f0_speaker_id", type=int)
    parser.add_argument("--f0_correct", type=float, default=0)
    parser.add_argument("--with_load", action="store_true")
    run(**vars(parser.parse_args()))
