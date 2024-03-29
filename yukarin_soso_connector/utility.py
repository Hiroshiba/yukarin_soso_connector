import re
from pathlib import Path
from typing import List

import torch


def extract_number(f):
    s = re.findall(r"\d+", str(f))
    return int(s[-1]) if s else -1


def get_predictor_model_path(
    model_dir: Path,
    iteration: str = None,
    prefix: str = "predictor_",
    postfix: str = ".pth",
):
    if iteration is None:
        paths = model_dir.glob(prefix + "*" + postfix)
        model_path = list(sorted(paths, key=extract_number))[-1]
    else:
        model_path = model_dir / (prefix + iteration + postfix)
        assert model_path.exists()
    return model_path


def remove_weight_norm(m):
    try:
        torch.nn.utils.remove_weight_norm(m)
    except ValueError:
        pass


def select_embedding(embedding: torch.nn.Embedding, indexes: List[int]):
    return torch.nn.Embedding(
        len(indexes), embedding.embedding_dim, _weight=embedding.weight[indexes]
    )
