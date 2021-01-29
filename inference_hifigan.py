import json
import os

import numpy
import torch
from hifi_gan.env import AttrDict
from hifi_gan.models import Generator


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


def inference_hifigan(x: numpy.ndarray, checkpoint_file, config_file):
    with open(config_file) as f:
        data = f.read()

    json_config = json.loads(data)
    h = AttrDict(json_config)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    generator = Generator(h).to(device)

    state_dict_g = load_checkpoint(checkpoint_file, device)
    generator.load_state_dict(state_dict_g["generator"])

    generator.eval()
    generator.remove_weight_norm()
    with torch.no_grad():
        x = torch.FloatTensor(x).unsqueeze(0).to(device)
        y_g_hat = generator(x)
        audio = y_g_hat.squeeze()
        audio = audio.cpu().numpy()

    return audio
