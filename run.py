import argparse
import shutil
from itertools import product
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import List, Optional

import soundfile
from tqdm import tqdm

from yukarin_soso_connector.forwarder import Forwarder


def run(
    yukarin_s_model_dir: Path,
    yukarin_sa_model_dir: Optional[Path],
    yukarin_saa_model_dir: Optional[Path],
    yukarin_soso_model_dir: Optional[Path],
    yukarin_sosoa_model_dir: Optional[Path],
    hifigan_model_dir: Path,
    hifigan_model_iteration: Optional[str],
    output_dir: Path,
    use_gpu: bool,
    texts: Optional[List[str]],
    text_path: Optional[Path],
    speaker_ids: List[int],
    f0_speaker_ids: Optional[List[int]],
    length_speaker_ids: Optional[List[int]],
    f0_correct: float,
    prefix: Optional[str],
):
    output_dir.mkdir(parents=True, exist_ok=True)

    if text_path is not None:
        texts = [line.strip() for line in text_path.read_text().splitlines()]
    assert texts is not None

    forwarder = Forwarder(
        yukarin_s_model_dir=yukarin_s_model_dir,
        yukarin_sa_model_dir=yukarin_sa_model_dir,
        yukarin_saa_model_dir=yukarin_saa_model_dir,
        yukarin_soso_model_dir=yukarin_soso_model_dir,
        yukarin_sosoa_model_dir=yukarin_sosoa_model_dir,
        hifigan_model_dir=hifigan_model_dir,
        hifigan_model_iteration=hifigan_model_iteration,
        prefix=prefix,
        use_gpu=use_gpu,
    )

    if f0_speaker_ids is None:
        f0_speaker_ids = speaker_ids

    if length_speaker_ids is None:
        length_speaker_ids = f0_speaker_ids

    assert len(f0_speaker_ids) == len(speaker_ids)
    assert len(length_speaker_ids) == len(speaker_ids)

    for text, (speaker_id, f0_speaker_id, length_speaker_id) in tqdm(
        list(product(texts, zip(speaker_ids, f0_speaker_ids, length_speaker_ids)))
    ):
        wave, _ = forwarder.forward(
            text=text,
            speaker_id=speaker_id,
            f0_speaker_id=f0_speaker_id,
            length_speaker_id=length_speaker_id,
            f0_correct=f0_correct,
        )

        with NamedTemporaryFile(suffix=".wav") as tmp:
            soundfile.write(tmp.name, data=wave, samplerate=24000)
            shutil.copyfile(tmp.name, output_dir / f"{text}-{speaker_id}.wav")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--yukarin_s_model_dir", type=Path, required=True)
    parser.add_argument("--yukarin_sa_model_dir", type=Path)
    parser.add_argument("--yukarin_saa_model_dir", type=Path)
    parser.add_argument("--yukarin_soso_model_dir", type=Path)
    parser.add_argument("--yukarin_sosoa_model_dir", type=Path)
    parser.add_argument("--hifigan_model_dir", type=Path)
    parser.add_argument("--hifigan_model_iteration")
    parser.add_argument("--output_dir", type=Path, default=Path("./"))
    parser.add_argument("--use_gpu", action="store_true")
    parser.add_argument("--texts", nargs="+")
    parser.add_argument("--text_path", type=Path)
    parser.add_argument("--speaker_ids", nargs="+", type=int, required=True)
    parser.add_argument("--f0_speaker_ids", nargs="+", type=int)
    parser.add_argument("--length_speaker_ids", nargs="+", type=int)
    parser.add_argument("--f0_correct", type=float, default=0)
    parser.add_argument("--prefix")
    run(**vars(parser.parse_args()))
