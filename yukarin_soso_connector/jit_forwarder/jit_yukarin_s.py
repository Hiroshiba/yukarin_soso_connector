from typing import Optional

import torch
from torch import Tensor, nn
from yukarin_s.network.predictor import Predictor as YukarinSPredictor


class JitYukarinS(nn.Module):
    def __init__(self, predictor: YukarinSPredictor):
        super().__init__()
        self.predictor = torch.jit.script(predictor)
        print("--- yukarin_s predictor ---\n", self.predictor.code)

    def forward(self, phoneme_list: Tensor, speaker_id: Optional[Tensor]):
        if speaker_id is not None:
            speaker_id = speaker_id.reshape((1,)).to(torch.int64)

        output = self.predictor(
            phoneme_list=phoneme_list.unsqueeze(0), speaker_id=speaker_id
        )[0]
        return output
