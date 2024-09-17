from __future__ import annotations
# from math import pi, log

import torch
import torch.nn as nn
from torch.nn import Module
from torch.cuda.amp import autocast
from torch import einsum

from einops import rearrange, repeat

def rotate_half(x):
    x = rearrange(x, '... (d r) -> ... d r', r = 2)
    x1, x2 = x.unbind(dim = -1)
    x = torch.stack((-x2, x1), dim = -1)
    return rearrange(x, '... d r -> ... (d r)')

def apply_rotary_emb(freqs, t, start_index = 0, scale = 1., seq_dim = -2):
    dtype = t.dtype

    rot_dim = freqs.shape[-1]
    end_index = start_index + rot_dim

    assert rot_dim <= t.shape[-1], f'feature dimension {t.shape[-1]} is not of sufficient size to rotate in all the positions {rot_dim}'

    t_left, t, t_right = t[..., :start_index], t[..., start_index:end_index], t[..., end_index:]
    t = (t * freqs.cos()) + (rotate_half(t) * freqs.sin())
    out = torch.cat((t_left, t, t_right), dim = -1)

    return out.type(dtype)

class RotaryEmbedding(Module):
    def __init__(
        self,
        dim,
        scale,
        theta = 10000,
    ):
        super().__init__()

        self.theta = theta
        self.dim = dim
        assert self.dim % 4 == 0
        pos_dim = self.dim // 2
        self.freqs = nn.Parameter(1. / (self.theta ** (torch.arange(0, pos_dim, 2)[:(pos_dim // 2)].float() / pos_dim)), requires_grad=False)
        self.scale = scale

    @property
    def device(self):
        return self.dummy.device

    @autocast(enabled = False)
    def forward(self, t, pos):
        pos = self.scale * pos

        pos_x = pos[:, :, 0]
        pos_y = pos[:, :, 1]

        freq_x = self.encode_freqs(pos_x)
        freq_y = self.encode_freqs(pos_y)


        freqs = torch.cat([freq_x, freq_y], dim=-1)

        return  apply_rotary_emb(freqs, t)

    @autocast(enabled = False)
    def encode_freqs(
        self,
        pos,
    ):

        freqs = self.freqs
        freqs = einsum('..., f -> ... f', pos.type(freqs.dtype), freqs)
        freqs = repeat(freqs, '... n -> ... (n r)', r = 2)

        return freqs
