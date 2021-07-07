import re
from pathlib import Path

import torch


def extract_number(f):
    s = re.findall(r"\d+", str(f))
    return int(s[-1]) if s else -1


def get_predictor_model_path(
    model_dir: Path,
    iteration: int = None,
    prefix: str = "predictor_",
    postfix: str = ".pth",
):
    if iteration is None:
        paths = model_dir.glob(prefix + "*" + postfix)
        model_path = list(sorted(paths, key=extract_number))[-1]
    else:
        model_path = model_dir / (prefix + "{}.pth".format(iteration))
        assert model_path.exists()
    return model_path


def remove_weight_norm(m):
    try:
        torch.nn.utils.remove_weight_norm(m)
    except ValueError:
        pass
