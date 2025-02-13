import os
import sys
sys.path.append(os.getcwd())

import torch
import torch.nn as nn

from models.mixer import IntraRegionK
from models.cyclemlp import CycleBlock, CycleMLP
from models.chordmixer import ChordMixerBlock


def get_chordmixer():
    net = nn.Sequential(
        ChordMixerBlock(
            hidden_size=1024 * 4,
            layer_dropout=0,
            embedding_size=1024,
            track_size=64,
            mlp_dropout=0,
            prenorm='LN',
            norm='LN',
        ),
        ChordMixerBlock(
            hidden_size=1024 * 4,
            layer_dropout=0,
            embedding_size=1024,
            track_size=64,
            mlp_dropout=0,
            prenorm='LN',
            norm='LN',
        ),
        ChordMixerBlock(
            hidden_size=1024 * 4,
            layer_dropout=0,
            embedding_size=1024,
            track_size=64,
            mlp_dropout=0,
            prenorm='LN',
            norm='LN',
        )
    )

    return net


def get_cylemlp():
    dim = 1024
    mlp_ratio = 4
    qkv_bias = False
    qk_scale = None
    attn_drop = 0
    skip_lam = 1
    mlp_fn = CycleMLP
    block_dpr = 0

    net = nn.Sequential(
        CycleBlock(dim, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                      attn_drop=attn_drop, drop_path=block_dpr, skip_lam=skip_lam, mlp_fn=mlp_fn),
        CycleBlock(dim, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                      attn_drop=attn_drop, drop_path=block_dpr, skip_lam=skip_lam, mlp_fn=mlp_fn),
        CycleBlock(dim, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                      attn_drop=attn_drop, drop_path=block_dpr, skip_lam=skip_lam, mlp_fn=mlp_fn),
    )
    return  net

def get_wsi_mixer():
    hidden_dim = 1024
    k = 64
    scale = 512

    net = nn.Sequential(
        IntraRegionK(hidden_dim, k, k ** 1, use_pos=False, scale=scale),
        IntraRegionK(hidden_dim, k, k ** 2, use_pos=False, scale=scale),
        IntraRegionK(hidden_dim, k, k ** 3, use_pos=False, scale=scale),
    )

    return net

def main():
    N = 128 * 128
    inputs = torch.randn(1, N, 1024)
    # print(inputs)

    # net1 = get_wsi_mixer()
    # net1 = get_cylemlp()
    net1 = get_chordmixer()
    net1 = net1.eval()

    # inputs = inputs.view(1, 128, 128, -1)


    with torch.inference_mode():
        out = net1(inputs)

    inputs1 = inputs.clone()


    # inputs1 = inputs1.view(1, N, -1)
    inputs1[:, 5000, :] = 0
    # inputs1 = inputs1.view(1, 128, 128, -1)

    print('difference between inputs1 and inputs:', (inputs1 != inputs).sum())
    with torch.inference_mode():
        out1 = net1(inputs1)

    # print(out.shape, out1.shape)
    out = out.view(1, N, -1)
    out1 = out1.view(1, N, -1)
    diff = (out - out1).sum(dim=-1)
    # print((out - out1).sum(dim=-1))
    print('diff: ', diff.shape)
    print(diff)
    print('diff sum:', (diff == 0).sum())


if __name__ == '__main__':
    main()
