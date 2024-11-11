import os
import sys
import random

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
    print(train_dataset)

    rotary_emb = RoPE(dim=1024)

    my_rotary = MyRoPE(dim=1024, scale=1)



    # freqs = torch.randn(10, 2)
    # # print(freqs)
    # # freqs = repeat(freqs, '... n -> ... (n r)', r = 2)
    # # print(freqs)
    # # print(freqs.shape)
    # # import sys; sys.exit()
    angles = torch.tensor([])
    for feat, pos, label in train_dataset:

        feat = feat.unsqueeze(dim=0)
        pos = pos.unsqueeze(dim=0)

        diff = compute_distance(pos)
        theta = compute_distance(pos)
        angle = diff * theta
        # angle.append(angle)
        # print(angle.shape)
        angles = torch.cat([angles, angle], dim=-1)
        # print(angles.shape)
        continue
    # weight, density = torch.histogram(angles[0] / 2 * 3.14, density=False, bins=30)
    # print(weight)
    # print(density)
    print('????')
    plt.hist(angles[0] / 2 * 3.14, bins=30, color='skyblue', edgecolor='black')
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.title('Basic Histogram')
    # plt.savefig('', **kwargs)
    plt.savefig('simple_plot.jpg', dpi=300, bbox_inches='tight')




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
