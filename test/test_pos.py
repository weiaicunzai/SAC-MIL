import os
import sys
import random
import math

sys.path.append(os.getcwd())
from einops import rearrange, repeat


import torch
from rotary_embedding_torch.rotary_embedding_torch import RotaryEmbedding as RoPE
from dataset.wsi_dataset import WSIDataset
from models.pos import RotaryEmbedding as MyRoPE
import matplotlib.pyplot as plt


# from pos_test import RotaryEmbedding as MyRoPE

def compute_distance(pos):
    diff = torch.norm(pos, p=2, dim=-1)
    return diff

def compute_theta(pos):
    theta = torch.atan2(pos[:, :, 0], pos[:, :, 1])
    return theta

if __name__ == '__main__':
    data_root = '/data/smb/syh/WSI_cls/cam16'
    train_dataset = WSIDataset(data_root, state='train', fold=0)
    # print(train_dataset)

    rotary_emb = RoPE(dim=1024)

    my_rotary = MyRoPE(dim=1024, scale=1)



    # freqs = torch.randn(10, 2)
    # # print(freqs)
    # # freqs = repeat(freqs, '... n -> ... (n r)', r = 2)
    # # print(freqs)
    # # print(freqs.shape)
    # # import sys; sys.exit()
    theta = 10000
    dim = 1024
    freqs = 1. / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))

    freqs = freqs.unsqueeze(dim=0)

    angles = torch.tensor([])
    angles = []
    count = 0
    for feat, pos, label in train_dataset:

        count +=1
        if count > 20:
            break

        feat = feat.unsqueeze(dim=0)
        pos = pos.unsqueeze(dim=0)

        radius = compute_distance(pos)
        theta = compute_theta(pos)
        # angle = diff * theta
        # angle.append(angle)
        # print(angle.shape)
        # angles = torch.cat([angles, angle], dim=-1)
        # print(angles.shape)
        # continue
        # radius.unsqueeze(dim=-1) * freqs.unsq
        radius = radius / math.sqrt(2) * 1024 * 4
        print(radius.unsqueeze(dim=-1).shape, freqs.unsqueeze(dim=0).shape )
        angle = torch.einsum('bji,bik->bjk', radius.unsqueeze(dim=-1), freqs.unsqueeze(dim=0))
        # result = torch.einsum('ijk,ijk->ij', , B)
        # print(radius, radius.shape)
        # print(angle.shape)
        # print(angle)
        # angles.append(freqs.view(-1))
        # angles.append(radius.view(-1))
        # angles.append(freqs.view(-1))
        angles.append(angle[:, :, 500].view(-1))

        # for i in freqs.view(-1):
        #     print(i.item())

        # import sys; sys.exit()
        # break

    # print(angles)


    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    # 假设 data 是一个包含上亿个浮点数的数组
    # data = np.random.randn(100000000)

    data = torch.cat(angles) % (math.pi * 2)


    # x = np.arange(len(data))

    # print(x.shape)

    # # 绘制点
    # plt.scatter(x, data, color='blue', marker='o')

    # # 设置图表标题和标签
    # plt.title('Scatter Plot of One-Dimensional Array')
    # plt.xlabel('Index')
    # plt.ylabel('Value')
    # plt.grid(True)
    # plt.savefig('test1.jpg')

    # import sys; sys.exit()
    # 转换为 Pandas Series
    series = pd.Series(data)
    # series = pd.Series(data)

    # 绘制直方图
    series.plot.hist(bins=100, range=(0, 2 * math.pi), color='blue', alpha=0.7, edgecolor='black')
    plt.title('Frequency Distribution of Float Numbers')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.grid(True)
    print('???')
    # plt.show()
    # plt.hist(data, bins=50,  density=False, color='blue', alpha=0.7, edgecolor='black')
    # plt.title('Normalized Frequency Distribution of Float Numbers')
    # plt.xlabel('Value')
    # plt.ylabel('Density')
    # plt.grid(True)
    plt.savefig('test1.jpg')
    # weight, density = torch.histogram(angles[0] / 2 * 3.14, density=False, bins=30)
    # print(weight)
    # print(density)
    # print('????')
    # plt.hist(angles[0] / 2 * 3.14, bins=30, color='skyblue', edgecolor='black')
    # plt.xlabel('Values')
    # plt.ylabel('Frequency')
    # plt.title('Basic Histogram')
    # # plt.savefig('', **kwargs)
    # plt.savefig('simple_plot.jpg', dpi=300, bbox_inches='tight')




    import sys; sys.exit()


        # print(feat.shape, pos.shape, label)
        # print(feat[0].mean())
        # print(feat.shape, pos.shape)
#        feat1 = rotary_emb.rotate_queries_or_keys(feat, pos[:, :, 0])
#        feat1 = feat1[:, :, :512]
#        # print(feat1)
#        seq_len = feat1.shape[1]
#        dim = feat1.shape[2]
#
#        myfeat = my_rotary(feat, pos)
#        myfeat = myfeat[:, :, :512]
#
#        seq_idx = random.randint(0, seq_len)
#        dim_idx = random.randint(0, dim)
#        # print(myfeat.shape, feat1.shape)
#
#        if myfeat[0, seq_idx, dim_idx] != feat1[0, seq_idx, dim_idx]:
#            print(feat1.shape, myfeat.shape)
#            print('mean:', (myfeat - feat1).mean())
#            print(myfeat[0, seq_idx, dim_idx] - feat1[0, seq_idx, dim_idx], myfeat[0, seq_idx, dim_idx], feat1[0, seq_idx, dim_idx])
#            import sys; sys.exit()
