import os
import sys
sys.path.append(os.getcwd())

import csv
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

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
    # N = 128 * 128
    N = 100 * 100
    inputs = torch.randn(1, N, 1024)
    # print(inputs)

    # net1 = get_wsi_mixer()
    # net1 = get_cylemlp()
    net1 = get_chordmixer()
    net1 = net1.eval()

    # inputs = inputs.view(1, 128, 128, -1)#只有第二个要用这个

    with torch.inference_mode():
        out = net1(inputs)

    inputs1 = inputs.clone()

    # inputs1 = inputs1.view(1, N, -1) #只有第二个要用这个
    half_length = inputs1.shape[1] // 2
    inputs1[:, half_length, :] = 0
    # inputs1 = inputs1.view(1, 128, 128, -1)#只有第二个要用这个

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

inputs=[]
def get_net1():
    N = 128 * 128
    # inputs1=torch.randn(1,N,1024)
    net = get_wsi_mixer()
    net = net.cuda()
    print(net)
    net = net.eval()
    # diff1 = torch.zeros(1, 16384)  #wsi_mixer
    diff1 = 0
    res = []
    for i,inputs1 in enumerate(inputs):
        print('1:',i)
        inputs1 = inputs1.cuda()
        with torch.inference_mode():
            out1 = net(inputs1)
        inputs2=inputs1.clone()
        inputs2[:, 5000, :] = 0
        with torch.inference_mode():
            out2 = net(inputs2)
        # diff = (out1-out2).sum(dim=-1)
        # diff=diff.view(-1)
        diff = out1 - out2
        # diff=torch.abs(diff)
        # print('before', diff, diff.max(), diff.std())
        # diff = diff / diff.max()
        # diff = diff.norm2()
        # l2_norm = torch.norm(diff, dim=-1, p='fro', keepdim=True)
        diff = torch.norm(diff, dim=-1, p=2, keepdim=True)
        diff = diff.view(-1)
        # print(diff.shape)
        # print('after', l2_norm.max(), l2_norm.min() )
        # diff = diff / l2_norm
        # diff1+=diff
        diff1+= diff
        # res.append(diff)
        # print(diff1.shape)
    diff1 = diff1 / len(inputs)
    return diff1

def get_net2():
    N = 128 * 128
    # inputs1=torch.randn(1,N,1024)
    net = get_cylemlp()
    net = net.eval().cuda()
    # diff2 = torch.zeros(1, 16384)  #cylemlp
    diff2 = 0
    for i,inputs1 in enumerate(inputs):
        print('2:',i)
        inputs1 = inputs1.view(1, 128, 128, -1).cuda()#变成四维张量
        with torch.inference_mode():
            out1 = net(inputs1)
        inputs2=inputs1.clone()
        inputs2 = inputs2.view(1, N, -1) #再变回三维进行数据处理
        inputs2[:, 5000, :] = 0
        inputs2 = inputs2.view(1, 128, 128, -1)#再将处理之后的变为四维
        with torch.inference_mode():
            out2 = net(inputs2)
        # diff = (out1-out2).sum(dim=-1)
        diff = torch.norm(out1-out2, dim=-1, p=1, keepdim=True)
        diff=diff.view(-1)
        diff=torch.abs(diff)
        diff2+=diff
    diff2 = diff2 / len(inputs)
    return diff2

def get_net3():
    N = 128 * 128
    # inputs1=torch.randn(1,N,1024)
    net = get_chordmixer()
    net = net.eval().cuda()
    # diff3 = torch.zeros(1, 16384)  #chordmixer
    diff3 = 0
    for i,inputs1 in enumerate(inputs):
        print('3:',i)
        inputs1 = inputs1.cuda()
        with torch.inference_mode():
            out1 = net(inputs1)
        inputs2=inputs1.clone()
        inputs2[:, 5000, :] = 0 #第5000个位置为0，也就是有1024个元素被置0
        with torch.inference_mode():
            out2 = net(inputs2)
        # diff = (out1-out2).sum(dim=-1)
        diff = torch.norm(out1-out2, dim=-1, p=1, keepdim=True)

        diff=diff.view(-1)
        diff=torch.abs(diff)
        print(diff.shape)
        diff3+=diff

    diff3 = diff3 / len(inputs)
    return diff3

if __name__ == '__main__':
    # main()
    N = 128 * 128
    total=10
    for i in range(total):
        torch.manual_seed(i)
        input=torch.randn(1,N,1024)
        inputs.append(input)
    diff1=get_net1()
    # print(diff1)
    diff2=get_net2()
    diff3=get_net3()
    # print(diff1,diff2,diff3)
    list1=diff1.tolist()
    list2=diff2.tolist()
    list3=diff3.tolist()
    # print(list1)
    # print(list2)
    # print(list3)
    # fig, axs = plt.subplots(1, 3)
    # axs[0].bar(range(len(list1)), list1, color='skyblue')
    # axs[0].set_title('(a)')
    # axs[1].bar(range(len(list2)), list2, color='lightgreen')
    # axs[1].set_title('(b)')
    # axs[2].bar(range(len(list3)), list3, color='salmon')
    # axs[2].set_title('(c)')
    # plt.tight_layout()
    # plt.show()

    with open('mytest/data1.csv','w',newline='') as file:
        writer=csv.writer(file)
        # for row in zip(list1[0],list2[0],list3[0]):
        for row in zip(list1,list2,list3):
            writer.writerow(row)
