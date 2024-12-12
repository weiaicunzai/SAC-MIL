from __future__ import annotations
from math import pi, log, sqrt

import torch
from torch.amp import autocast
from torch.nn import Module, ModuleList
from torch import nn, einsum, broadcast_tensors, Tensor

from einops import rearrange, repeat

from typing import Literal

# helper functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

# broadcat, as tortoise-tts was using it

def broadcat(tensors, dim = -1):
    broadcasted_tensors = broadcast_tensors(*tensors)
    return torch.cat(broadcasted_tensors, dim = dim)

def slice_at_dim(t, dim_slice: slice, *, dim):
    dim += (t.ndim if dim < 0 else 0)
    colons = [slice(None)] * t.ndim
    colons[dim] = dim_slice
    return t[tuple(colons)]

# rotary embedding helper functions

def rotate_half(x):
    x = rearrange(x, '... (d r) -> ... d r', r = 2)
    x1, x2 = x.unbind(dim = -1)
    x = torch.stack((-x2, x1), dim = -1)
    return rearrange(x, '... d r -> ... (d r)')

@autocast('cuda', enabled = False)
def apply_rotary_emb(
    freqs,
    t,
    start_index = 0,
    scale = 1.,
    seq_dim = -2,
    freqs_seq_dim = None
):
    dtype = t.dtype

    if not exists(freqs_seq_dim):
        if freqs.ndim == 2 or t.ndim == 3:
            freqs_seq_dim = 0

    if t.ndim == 3 or exists(freqs_seq_dim):
        seq_len = t.shape[seq_dim]
        freqs = slice_at_dim(freqs, slice(-seq_len, None), dim = freqs_seq_dim)

    rot_dim = freqs.shape[-1]
    end_index = start_index + rot_dim

    assert rot_dim <= t.shape[-1], f'feature dimension {t.shape[-1]} is not of sufficient size to rotate in all the positions {rot_dim}'

    # Split t into three parts: left, middle (to be transformed), and right
    t_left = t[..., :start_index]
    t_middle = t[..., start_index:end_index]
    t_right = t[..., end_index:]

    # Apply rotary embeddings without modifying t in place    
    t_transformed = (t_middle * freqs.cos() * scale) + (rotate_half(t_middle) * freqs.sin() * scale)
        
    out = torch.cat((t_left, t_transformed, t_right), dim=-1)

    return out.type(dtype)

# learned rotation helpers

def apply_learned_rotations(rotations, t, start_index = 0, freq_ranges = None):
    if exists(freq_ranges):
        rotations = einsum('..., f -> ... f', rotations, freq_ranges)
        rotations = rearrange(rotations, '... r f -> ... (r f)')

    rotations = repeat(rotations, '... n -> ... (n r)', r = 2)
    return apply_rotary_emb(rotations, t, start_index = start_index)

# classes

class RotaryEmbedding(Module):
    def __init__(
        self,
        dim,
        custom_freqs: Tensor | None = None,
        freqs_for:  Literal['lang', 'pixel', 'constant'] = 'lang',
        theta = 10000,
        max_freq = 10,
        num_freqs = 1,
        learned_freq = False,
        # use_xpos = False,
        # xpos_scale_base = 512,
        theta_rescale_factor = 1.,
        # seq_before_head_dim = False,
        # cache_if_possible = True,
        # cache_max_seq_len = 8192,
        scale=1,
        mod=None
    ):
        super().__init__()
        # proposed by reddit user bloc97, to rescale rotary embeddings to longer sequence length without fine-tuning
        # has some connection to NTK literature
        # https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/

        theta *= theta_rescale_factor ** (dim / (dim - 2))

        self.freqs_for = freqs_for

        if exists(custom_freqs):
            freqs = custom_freqs
        elif freqs_for == 'lang':
            freqs = 1. / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
        elif freqs_for == 'pixel':
            freqs = torch.linspace(1., max_freq / 2, dim // 2) * pi
        elif freqs_for == 'constant':
            freqs = torch.ones(num_freqs).float()

        # self.cache_if_possible = cache_if_possible
        # self.cache_max_seq_len = cache_max_seq_len

        # self.register_buffer('cached_freqs', torch.zeros(cache_max_seq_len, dim), persistent = False)
        # self.register_buffer('cached_freqs_seq_len', torch.tensor(0), persistent = False)

        self.freqs = nn.Parameter(freqs, requires_grad = learned_freq)

        # self.learned_freq = learned_freq

        # dummy for device

        # self.register_buffer('dummy', torch.tensor(0), persistent = False)

        # default sequence dimension

        # self.seq_before_head_dim = seq_before_head_dim
        # self.default_seq_dim = -3 if seq_before_head_dim else -2

        # interpolation factors

        # assert interpolate_factor >= 1.
        # self.interpolate_factor = interpolate_factor

        # xpos

        # self.use_xpos = use_xpos

        # if not use_xpos:
            # return

        # scale = (torch.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)
        # self.scale_base = xpos_scale_base

        # self.register_buffer('scale', scale, persistent = False)
        # self.register_buffer('cached_scales', torch.zeros(cache_max_seq_len, dim), persistent = False)
        # self.register_buffer('cached_scales_seq_len', torch.tensor(0), persistent = False)

        # add apply_rotary_emb as static method

        # self.apply_rotary_emb = staticmethod(apply_rotary_emb)

        self.scale = scale
        self.mod = mod
        print('mod: ', mod)

    def compute_radius(self, pos):
        diff = torch.norm(pos, p=2, dim=-1)
        diff = diff / sqrt(2) * self.scale
        assert diff.max() <= self.scale
        return diff

    def compute_theta(self, pos):
         theta = torch.atan2(pos[:, :, 0], pos[:, :, 1])
         return theta

    # @property
    # def device(self):
    #     return self.dummy.device

    # def get_seq_pos(self, seq_len, device, dtype, offset = 0):
    #     return (torch.arange(seq_len, device = device, dtype = dtype) + offset) / self.interpolate_factor

    def rotate_queries_or_keys(self, t, pos, seq_dim = -2):
    # def rotate_queries_or_keys(self, t, seq_dim = None, offset = 0, scale = None):
        # seq_dim = default(seq_dim, self.default_seq_dim)

        # assert not self.use_xpos or exists(scale), 'you must use `.rotate_queries_and_keys` method instead and pass in both queries and keys, for length extrapolatable rotary embeddings'

        # device, dtype, seq_len = t.device, t.dtype, t.shape[seq_dim]

        radius = self.compute_radius(pos)
# 
        theta = self.compute_theta(pos)

        # scale = 512
        assert radius.max() <= self.scale
        freqs = self.forward(radius)
        freqs = freqs + theta.unsqueeze(dim=-1)

        if seq_dim == -3:
            freqs = rearrange(freqs, 'n d -> n 1 d')

        #return apply_rotary_emb(freqs, t, scale = default(scale, 1.), seq_dim = seq_dim)
        return apply_rotary_emb(freqs, t, scale = 1, seq_dim = seq_dim)

    def rotate_angle(self, pos):
        radius = self.compute_radius(pos)
        theta = self.compute_theta(pos)

        freqs = self.freqs

        freqs = einsum('..., f -> ... f', radius.type(freqs.dtype), freqs)
        freqs = repeat(freqs, '... n -> ... (n r)', r = 2)

        angles = freqs + theta.unsqueeze(dim=-1)

        return angles


    @autocast('cuda', enabled = False)
    def forward(
        self,
        x,
        pos,
    ):
        angles = self.rotate_angle(pos)

        if self.mode:
            angles = angles % self.mod


        return apply_rotary_emb(angles, x, scale = 1, seq_dim = -2)





    # @autocast('cuda', enabled = False)
    # def forward(
    #     self,
    #     t: Tensor,
    #     # seq_len = None,
    #     # offset = 0
    # ):
    #     # should_cache = (
    #     #     self.cache_if_possible and
    #     #     not self.learned_freq and
    #     #     exists(seq_len) and
    #     #     self.freqs_for != 'pixel' and
    #     #     (offset + seq_len) <= self.cache_max_seq_len
    #     # )

    #     # if (
    #     #     should_cache and \
    #     #     exists(self.cached_freqs) and \
    #     #     (offset + seq_len) <= self.cached_freqs_seq_len.item()
    #     # ):
    #     #     return self.cached_freqs[offset:(offset + seq_len)].detach()

    #     freqs = self.freqs

    #     freqs = einsum('..., f -> ... f', t.type(freqs.dtype), freqs)
    #     freqs = repeat(freqs, '... n -> ... (n r)', r = 2)

    #     return freqs