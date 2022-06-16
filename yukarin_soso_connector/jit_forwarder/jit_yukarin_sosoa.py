from typing import List, Optional

import torch
from espnet_pytorch_library.tacotron2.decoder import Postnet
from torch import Tensor, nn
from torch.nn.utils.rnn import pad_sequence
from yukarin_sosoa.network.predictor import Predictor as YukarinSosoaPredictor


def make_pad_mask(lengths: Tensor):
    bs = lengths.shape[0]
    maxlen = lengths.max()

    seq_range = torch.arange(0, maxlen, dtype=torch.int64, device=lengths.device)
    seq_range_expand = seq_range.unsqueeze(0).expand(bs, maxlen)
    seq_length_expand = lengths.unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand
    return mask


@torch.jit.script
def make_non_pad_mask(lengths: Tensor):
    return ~make_pad_mask(lengths)


class JitPostnet(nn.Module):
    def __init__(self, net: Postnet):
        super().__init__()
        self.postnet = net.postnet

    def forward(self, xs):
        for net in self.postnet:
            xs = net(xs)
        return xs


class JitYukarinSosoa(nn.Module):
    def __init__(self, predictor: YukarinSosoaPredictor, device):
        super().__init__()

        predictor.encoder.embed[0].pe = predictor.encoder.embed[0].pe.to(device)

        self.speaker_embedder = torch.jit.script(predictor.speaker_embedder)
        self.pre = torch.jit.script(predictor.pre)
        self.encoder = torch.jit.trace(
            predictor.encoder,
            (
                torch.rand(1, 1, predictor.pre.out_features, device=device),
                make_non_pad_mask(torch.tensor([1], device=device)).unsqueeze(-2),
            ),
        )
        self.post = torch.jit.script(predictor.post)
        self.postnet = torch.jit.script(JitPostnet(predictor.postnet))

    def forward(
        self,
        f0: Tensor,
        phoneme: Tensor,
        speaker_id: Tensor,
    ):
        h = torch.cat((f0, phoneme), dim=1)  # (length, ?)
        h = h.unsqueeze(0)  # (batch_size, length, ?)

        if self.speaker_embedder is not None and speaker_id is not None:
            speaker_id = self.speaker_embedder(speaker_id)
            speaker_id = speaker_id.unsqueeze(dim=1)  # (batch_size, 1, ?)
            speaker_feature = speaker_id.expand(
                speaker_id.shape[0], h.shape[1], speaker_id.shape[2]
            )  # (batch_size, length, ?)
            h = torch.cat((h, speaker_feature), dim=2)  # (batch_size, length, ?)

        h = self.pre(h)

        mask = torch.ones(h.shape[:2], device=h.device)
        h, _ = self.encoder(h, mask)

        output1 = self.post(h)
        output2 = output1 + self.postnet(output1.transpose(1, 2)).transpose(1, 2)
        return output2[0]
