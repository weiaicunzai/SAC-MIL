import math

import torch

# import models.configs as configs

# from os.path import join as pjoin

from torch import nn
from torch.nn.modules.utils import _pair
import torch.nn.functional as F

from models.Pos import RotaryEmbedding


TOK_FC_0 = "token_mixing/Dense_0"
TOK_FC_1 = "token_mixing/Dense_1"
CHA_FC_0 = "channel_mixing/Dense_0"
CHA_FC_1 = "channel_mixing/Dense_1"
PRE_NORM = "LayerNorm_0"
POST_NORM = "LayerNorm_1"


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


class MlpBlock(nn.Module):
    def __init__(self, hidden_dim, ff_dim):
        super(MlpBlock, self).__init__()
        self.fc0 = nn.Linear(hidden_dim, ff_dim, bias=True)
        self.fc1 = nn.Linear(ff_dim, hidden_dim, bias=True)
        self.act_fn = nn.GELU()

    def forward(self, x):
        x = self.fc0(x)
        x = self.act_fn(x)
        x = self.fc1(x)
        return x


class MixerBlock(nn.Module):
    def __init__(self, hidden_dim, num_tokens, token_dim, channel_dim):
        super(MixerBlock, self).__init__()
        self.token_mlp_block = MlpBlock(num_tokens, token_dim)
        self.channel_mlp_block = MlpBlock(hidden_dim, channel_dim)
        self.pre_norm = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.post_norm = nn.LayerNorm(hidden_dim, eps=1e-6)

    def forward(self, x):
        h = x
        x = self.pre_norm(x)
        x = x.transpose(-1, -2)
        x = self.token_mlp_block(x)
        x = x.transpose(-1, -2)
        x = x + h

        h = x
        x = self.post_norm(x)
        x = self.channel_mlp_block(x)
        x = x + h
        return x

class ReducedMlpBlock(nn.Module):
    def __init__(self, reduced_token_dim, token_dim):
        super().__init__()
        self.fc0 = nn.Linear(reduced_token_dim, 1, bias=True)
        self.fc1 = nn.Linear(token_dim, token_dim, bias=True)
        self.act_fn = nn.GELU()

    def forward(self, x):
        x = self.fc0(x).squeeze(dim=-1)
        x = self.act_fn(x)
        x = self.fc1(x)
        return x

class ReducedMixerBlock(nn.Module):
    def __init__(self, wsi_len, num_tokens, hidden_dim, channel_dim):
        super().__init__()
        #self.token_mlp_block = MlpBlock(config.n_patches, config.tokens_mlp_dim)
        #self.channel_mlp_block = MlpBlock(config.hidden_dim, config.channels_mlp_dim)
        reduced_token_dim = wsi_len // num_tokens
        self.token_mlp_block = ReducedMlpBlock(reduced_token_dim, num_tokens)
        self.channel_mlp_block = MlpBlock(hidden_dim, channel_dim)
        self.pre_norm = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.post_norm = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.reduced_token_dim = reduced_token_dim

    def forward(self, x):
        B, L, D = x.shape
        h = x
        h = h.view(B, L // self.reduced_token_dim, self.reduced_token_dim, D)
        h = h.mean(dim=-2)
        # print(x.shape)
        x = self.pre_norm(x)
        x = x.transpose(-1, -2)
        # print(x.shape)
        # print(self.reduced_token_dim)
        x = x.view(B, D, L // self.reduced_token_dim, self.reduced_token_dim)
        x = self.token_mlp_block(x)
        x = x.transpose(-1, -2)
        # print(x.shape, h.shape)
        x = x + h

        h = x
        x = self.post_norm(x)
        x = self.channel_mlp_block(x)
        x = x + h
        return x


################################################################

class Classifier_1fc(nn.Module):
    def __init__(self, n_channels, n_classes, droprate=0.0):
        super(Classifier_1fc, self).__init__()
        self.fc = nn.Linear(n_channels, n_classes)
        self.droprate = droprate
        if self.droprate != 0.0:
            self.dropout = torch.nn.Dropout(p=self.droprate)

    def forward(self, x):

        if self.droprate != 0.0:
            x = self.dropout(x)
        x = self.fc(x)
        return x


class residual_block(nn.Module):
    def __init__(self, nChn=512):
        super(residual_block, self).__init__()
        self.block = nn.Sequential(
                nn.Linear(nChn, nChn, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(nChn, nChn, bias=False),
                nn.ReLU(inplace=True),
            )
    def forward(self, x):
        tt = self.block(x)
        x = x + tt
        return x

class DimReduction(nn.Module):
    def __init__(self, n_channels, m_dim=512, numLayer_Res=0):
        super(DimReduction, self).__init__()
        self.fc1 = nn.Linear(n_channels, m_dim, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.numRes = numLayer_Res

        self.resBlocks = []
        for ii in range(numLayer_Res):
            self.resBlocks.append(residual_block(m_dim))
        self.resBlocks = nn.Sequential(*self.resBlocks)

    def forward(self, x):

        x = self.fc1(x)
        x = self.relu1(x)

        if self.numRes > 0:
            x = self.resBlocks(x)

        return x


class Attention_Gated(nn.Module):
    def __init__(self, L=512, D=128, K=1):
        super(Attention_Gated, self).__init__()

        self.L = L
        self.D = D
        self.K = K

        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )

        self.attention_weights = nn.Linear(self.D, self.K)

    def forward(self, x):
        ## x: N x L
        A_V = self.attention_V(x)  # NxD
        A_U = self.attention_U(x)  # NxD
        A = self.attention_weights(A_V * A_U) # NxK
        # print(A.shape, 111)
        # A = torch.transpose(A, 1, 0)  # KxN

        # if isNorm:
        # print(A.shape)
        A = F.softmax(A, dim=-2)  # softmax over N
        # print(A[0, 0, :, 0].sum())

        return A  ### K x N


# class Attention_with_Classifier(nn.Module):
#     def __init__(self, L=512, D=128, K=1, num_cls=2, droprate=0):
#         super(Attention_with_Classifier, self).__init__()
#         self.attention = Attention_Gated(L, D, K)
#         self.classifier = Classifier_1fc(L, num_cls, droprate)
#     def forward(self, x): ## x: N x L
#         AA = self.attention(x)  ## K x N
#         afeat = torch.mm(AA, x) ## K x L
#         pred = self.classifier(afeat) ## K x num_cls
#         return pred

class InterRegion(nn.Module):

    # def __init__(self, )
    def __init__(self, hidden_dim, k):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.k = k
        self.inter_fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.intra_fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

    def unroll(self, x):
        D = x.shape[-1]
        x = x.split(
            D // self.k,
            dim=-1
        )

        res = []
        for idx, chunk in enumerate(x):
            chunk = torch.roll(chunk, -idx, dims=-2)
            res.append(chunk)

        res = torch.cat(res, dim=-1)

        return res

    def cal_region(self, x):
        centroids = x[:, :, :1, :]
        rest = x[:, :, 1:, :]

        num_set = int(centroids.shape[1] / self.k) * self.k

        left = centroids[:, :num_set, :, :]

        # print(left.shape) [1, 128, 1, 512]
        B, L, T, D = left.shape
        left = left.view(B, -1, self.k, D)
        left = self.roll(left)
        left = self.inter_fc(left)
        left = self.unroll(left)
        left = self.intra_fc(left)
        left = left.view(B, L, T, D)


        # print(left.shape, centroids.shape, num_set, self.k)
        if centroids.shape[1] != num_set * self.k:
            right = centroids[:, num_set:, :]
            assert right.shape[1] < self.k

            right = self.roll(right)
            right = self.inter_fc(right)
            right = self.unroll(right)
            right = self.intra_fc(right)

            centroids = torch.cat([left, right], dim=1)

        else:
            centroids = left

        x = torch.cat([centroids, rest], dim=2)

        return x

    def roll(self, x):
        D = x.shape[-1]
        x = x.split(
            D // self.k,
            dim=-1
        )

        res = []
        for idx, chunk in enumerate(x):
            # print(chunk.shape, idx)
            # print(chunk[0, 0, :, 0])
            chunk = torch.roll(chunk, idx, dims=-2)
            # print(chunk[0, 0, :, 0])
            # print(idx)
            res.append(chunk)

        res = torch.cat(res, dim=-1)
        # print(res[0, 0, 3, 8*3: 8 * 4], tmp)

        return res

    def forward(self, x):

        index = 1
        B, L, D = x.shape
        x = x.view(B, L // self.k, self.k, D)
        padding = 0

        while(x.shape[1] >= self.k):

            x = self.cal_region(x)

            # restore one dimension
            x = x.view(B, -1, D)
            x = x[:, :L, :]

            # pad more
            index += 1
            padding = self.k ** index - L % self.k ** index
            if padding != 0:
                x = torch.cat([x, x[:, :padding, :]], dim=1)

            x = x.view(B, -1, self.k ** index, D)

        total_len = L
        centroids = x[:, :, :1, :]
        rest = x[:, :, 1:, :]
        B, L, R, D = centroids.shape
        centroids = centroids.view(B, L, D)
        centroids = self.roll(centroids)
        centroids = self.inter_fc(centroids)
        centroids = self.unroll(centroids)
        centroids = self.intra_fc(centroids)
        centroids = centroids.view(B, L, R, D)
        x = torch.cat([centroids, rest], dim=2)

        # restore one dimension
        x = x.view(B, -1, D)
        x = x[:, :total_len, :]

        # print(x.shape)

        return x



class IntraRegion(nn.Module):
    def __init__(self, hidden_dim, k):
        super().__init__()

        self.k = k
        self.inter_fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.intra_fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
        #     # nn.ReLU()
        )

        self.pre_norm = nn.LayerNorm(hidden_dim)
        # self.norm = nn.LayerNorm(hidden_dim)
        # self.relu = nn.ReLU()

    def forward(self, x):

        B, L, D = x.shape
        res = x
        x = self.pre_norm(x)
        x = x.view(B, L // self.k, self.k, D)
        # tmp = x
        x = self.roll(x)
        x = self.inter_fc(x)
        x = self.unroll(x)
        x = self.intra_fc(x)
        # x = self.relu(x)
        # print((x - tmp).sum())

        x = x.view(B, L, D)

        x = x + res
        # x = self.norm(x)

        return x

    def unroll(self, x):
        D = x.shape[-1]
        x = x.split(
            D // self.k,
            dim=-1
        )

        res = []
        for idx, chunk in enumerate(x):
            chunk = torch.roll(chunk, -idx, dims=-2)
            res.append(chunk)

        res = torch.cat(res, dim=-1)

        return res

    def roll(self, x):
        D = x.shape[-1]
        x = x.split(
            D // self.k,
            dim=-1
        )

        res = []
        for idx, chunk in enumerate(x):
            chunk = torch.roll(chunk, idx, dims=-2)
            res.append(chunk)

        res = torch.cat(res, dim=-1)

        return res

class Interaction(nn.Module):
    def __init__(self, hidden_dim, k):
        super().__init__()

        self.k = k
        self.inter_fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.intra_fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.pre_norm = nn.LayerNorm(hidden_dim)
        # self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):

        B, L, D = x.shape
        res = x
        # x = self.pre_norm(x)

        x = x.view(B, L // self.k, self.k, D)
        # x = self.roll(x)
        # x = self.inter_fc(x)
        # x = self.unroll(x)
        x = self.intra_fc(x)

        x = x.view(B, L, D)

        # x = x + res
        # x = self.norm(x)

        return x

    def unroll(self, x):
        D = x.shape[-1]
        x = x.split(
            D // self.k,
            dim=-1
        )

        res = []
        for idx, chunk in enumerate(x):
            chunk = torch.roll(chunk, -idx, dims=-2)
            res.append(chunk)

        res = torch.cat(res, dim=-1)

        return res

    def roll(self, x):
        D = x.shape[-1]
        x = x.split(
            D // self.k,
            dim=-1
        )

        res = []
        for idx, chunk in enumerate(x):
            chunk = torch.roll(chunk, idx, dims=-2)
            res.append(chunk)

        res = torch.cat(res, dim=-1)

        return res

class Rope(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, len):
        print(len)

class MlpMixer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes=2):
        super().__init__()
        # self.zero_head = zero_head
        self.num_classes = num_classes
        # self.seq_len = seq_len
        # self.token_dim = token_dim

        # net_dim = 512
        # self.fc1 = DimReduction(input_dim, m_dim=hidden_dim)
        self.fc1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )

        # self.attention =  Attention_Gated(hidden_dim)

        # self.num_tokens = num_tokens
        # self.channel_dim = channel_dim
        self.k = 64

        self.intra_region = nn.Sequential(
            IntraRegionK(hidden_dim, self.k, self.k ** 1),
            IntraRegionK(hidden_dim, self.k, self.k ** 2),
            IntraRegionK(hidden_dim, self.k, self.k ** 3)
        )
        # self.intra_region1 = IntraRegionK(hidden_dim, self.k, region_size=self.k**1)
        # self.intra_region2 = IntraRegionK(hidden_dim, self.k, region_size=self.k**2)
        # self.intra_region3 = IntraRegionK(hidden_dim, self.k, region_size=self.k**3)

        # self.inter_region = InterRegion(hidden_dim, self.k)

        # self.attention2 = Attention_Gated(hidden_dim)
        # self.channel_mix = nn.Sequential(

        # self.layer2 = MlpBlock(
        #     hidden_dim=hidden_dim,
        #     ff_dim= int(hidden_dim * 1)
        # )

        # )
        # self.inter_fc = nn.Sequential(
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.ReLU(inplace=True),
        # )
        # self.intra_fc = nn.Sequential(
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.ReLU(inplace=True),
        # )

        self.fc2 = nn.Linear(hidden_dim, self.num_classes)

        # self.stem = nn.Conv2d(in_channels=3,
        #                     #   out_channels=config.hidden_dim,
        #                       kernel_size=patch_size,
        #                       stride=patch_size)
        # self.head = nn.Linear(hidden_dim, num_classes, bias=True)
        # self.pre_head_ln = nn.LayerNorm(hidden_dim, eps=1e-6)

        # self.layer = nn.ModuleList()
        # for _ in range(config.num_blocks):
        # for _ in range(2):
        # seq_len =
        # self.layer1 = ReducedMixerBlock(
        #     wsi_len=seq_len,
        #     num_tokens=num_tokens,
        #     channel_dim=channel_dim,
        #     hidden_dim=hidden_dim
        # )
        # self.layer2 = MixerBlock(
        #     hidden_dim=hidden_dim,
        #     num_tokens=num_tokens,
        #     token_dim=token_dim,
        #     channel_dim=channel_dim
        # )

        # self.layer.append(copy.deepcopy(layer))
        # self.last_norm = nn.LayerNorm(hidden_dim)
        # self.pre_norm = nn.LayerNorm(hidden_dim)

        self.pos_emb = RotaryEmbedding(hidden_dim, scale=1)

    def pad_k(self, x):
        # x = self.()
        length = x.shape[1]
        # print(length)
        pad_k = self.k - length % self.k
        # print(k)
        if pad_k > 0:
            x = torch.cat([x, x[:, :pad_k, :]], dim=1)

        return x

    # def intra_region(self, x):
    #     # # x = self.
    #     # x = self.inter_fc
    #     # x = self.unroll(x)
    #     # print(x.shape)
    #     B, L, D = x.shape
    #     x = x.view(B, L // self.k, self.k, D)
    #     x = self.roll(x)
    #     x = self.inter_fc(x)
    #     x = self.unroll(x)
    #     x = self.intra_fc(x)

    #     x = x.view(B, L, D)

    #     return x


    # def cal_region(self, x):
    #     centroids = x[:, :, :1, :]
    #     rest = x[:, :, 1:, :]

    #     num_set = int(centroids.shape[1] / self.k) * self.k

    #     left = centroids[:, :num_set, :, :]

    #     # print(left.shape) [1, 128, 1, 512]
    #     B, L, T, D = left.shape
    #     left = left.view(B, -1, self.k, D)
    #     left = self.roll(left)
    #     left = self.inter_fc(left)
    #     left = self.unroll(left)
    #     left = self.intra_fc(left)
    #     left = left.view(B, L, T, D)


    #     # print(left.shape, centroids.shape, num_set, self.k)
    #     if centroids.shape[1] != num_set * self.k:
    #         right = centroids[:, num_set:, :]
    #         assert right.shape[1] < self.k

    #         right = self.roll(right)
    #         right = self.inter_fc(right)
    #         right = self.unroll(right)
    #         right = self.intra_fc(right)

    #         centroids = torch.cat([left, right], dim=1)

    #     else:
    #         centroids = left

    #     x = torch.cat([centroids, rest], dim=2)

    #     return x

    #def inter_region(self, x):
    #    # tmp = x.shape
    #    # B, L, D = x.shape
    #    # tmp = x


    #    # right = x[:, ::self.k:, :]

    #    # left =
    #    # print(centroids.shape)

    #    # centroids[centr]
    #    index = 1
    #    # centroids = x[:, ::self.k ** index, :]
    #    B, L, D = x.shape
    #    x = x.view(B, L // self.k, self.k, D)
    #    padding = 0
    #    # real_le

    #    while(x.shape[1] >= self.k):

    #        x = self.cal_region(x)

    #        # restore one dimension
    #        x = x.view(B, -1, D)
    #        x = x[:, :L, :]

    #        # pad more
    #        index += 1
    #        padding = self.k ** index - L % self.k ** index
    #        if padding != 0:
    #            x = torch.cat([x, x[:, :padding, :]], dim=1)

    #        x = x.view(B, -1, self.k ** index, D)


    #    total_len = L
    #    centroids = x[:, :, :1, :]
    #    rest = x[:, :, 1:, :]
    #    # print(centroids.shape)
    #    B, L, R, D = centroids.shape
    #    centroids = centroids.view(B, L, D)
    #    centroids = self.roll(centroids)
    #    centroids = self.inter_fc(centroids)
    #    centroids = self.unroll(centroids)
    #    centroids = self.intra_fc(centroids)
    #    centroids = centroids.view(B, L, R, D)
    #    x = torch.cat([centroids, rest], dim=2)



    #    # restore one dimension
    #    x = x.view(B, -1, D)
    #    x = x[:, :total_len, :]

    #    # print(x.shape)

    #    return x





     # 44380
    # def forward(self, x, labels=None):
    def forward(self, **kwargs):

        x = kwargs['data'].float() #[B, n, 1024]

        pos = kwargs['pos']
        # label =

        # x = self.stem(x)
        # x = x.flatten(2)
        # x = x.transpose(-1, -2)

        # [1, 2, 3, 4, 5, 6]
        # [5, , 3, 4]
        # for block in self.layer:
            # x = block(x)
        # print(x.shape)
        # x = self.pad_k(x)
        # print(x.shape, 'cccc')

        # print(x.shape)
        # x = self.pad_k(x)
        # print(x.shape, 'after')

        x = self.fc1(x)

        # x = self.pos_emb(x, pos)

        # x = self.pad_k(x)

        # x = self.pre_norm(x)
        # print(x.shape)
        x = self.intra_region(x)

        # x = self.intra_region1(x)
        # print(x.shape)
        # x = self.intra_region2(x)
        # x = self.intra_region3(x)
        # x = x.view(B, self.num_tokens, L // self.num_tokens, D)
        # x = self.layer1(x)
        # x = self.layer2(x)
        # x = self.pre_head_ln(x)
        # x = torch.mean(x, dim=1)
        # logits = self.head(x)
        # print(x.shape)

        # x = self.intra_region(x)
        # x = self.inter_region(x)
        # tAA = self.attention(x)
        # x = x * self.attention(x)
        # tattFeats = torch.einsum('bls,bld->bs', x, tAA)  ### n x fs
        # tattFeats = x * tAA

        # print(x.shape, 'before')
        # print(x.shape, 'after')

        # x = x.view(B, L // self.k, self.k, D)
        # print(x.shape)

        # x = self.intra_region(x)
        # x = self.inter_region(x)


        # tattFeats = x

        # x = self.last_norm(x)


        x = x.mean(dim=-2)

        logits = self.fc2(x)

        Y_hat = torch.argmax(logits, dim=1)
        Y_prob = F.softmax(logits, dim = 1)
        results_dict = {'logits': logits, 'Y_prob': Y_prob, 'Y_hat': Y_hat}
        return results_dict




class IntraRegionK(nn.Module):
    def __init__(self, hidden_dim, k, region_size):
        super().__init__()

        self.k = k
        self.region_size = region_size
        self.scale = int(self.k ** (math.log(self.region_size, self.k) - 1))

        # print(self.scale)

        self.inter_fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.intra_fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
        #     # nn.ReLU()
        )

        self.pre_norm = nn.LayerNorm(hidden_dim)
        # self.norm = nn.LayerNorm(hidden_dim)
        # self.relu = nn.ReLU()

    def forward(self, x):

        # B, L, D = x.shape
        res = x
        x = self.pre_norm(x)
        # tmp = x
        x = self.roll_all(x)
        x = self.inter_fc(x)
        x = self.unroll_all(x)
        # print((x - tmp).mean())
        x = self.intra_fc(x)

        x = x + res

        return x

    def split_parts(self, n):
        if n < self.region_size:
            return n

        # print('total', n, 'region size', self.region_size)
        mod = n % self.region_size
        # remain = self.region_size - mod
        num_part1 = n - mod

        assert num_part1 > 0


        return num_part1

    def roll_all(self, x):
        num_tokens = x.shape[1]

        # num_part1 = num_tokens - num_tokens % self.region_size
        # print(num_part1)

        num_part1 = self.split_parts(num_tokens)

        part1 = x[:, :num_part1, :]
        part2 = x[:, num_part1:, :]

        part1 = self.roll(part1)
        part2 = self.roll(part2)

        return torch.cat([part1, part2], dim=-2)

    def unroll_all(self, x):
        num_tokens = x.shape[1]

        num_part1 = self.split_parts(num_tokens)

        part1 = x[:, :num_part1, :]
        part2 = x[:, num_part1:, :]

        part1 = self.unroll(part1)
        part2 = self.unroll(part2)

        return torch.cat([part1, part2], dim=-2)

    def unroll(self, x):
        if x.shape[-2] == 0:
            return x

        B, L, D = x.shape

        if L >= self.region_size:
            x = x.view(B, -1, self.region_size, D)

        x = x.split(
            D // self.k,
            dim=-1
        )

        res = []

        for idx, chunk in enumerate(x):
            chunk = torch.roll(chunk, -idx * self.scale, dims=-2)
            res.append(chunk)

        res = torch.cat(res, dim=-1)

        if L >= self.region_size:
            res = res.view(B, L, D)

        return res

    def roll(self, x):

        if x.shape[-2] == 0:
            return x

        B, L, D = x.shape

        if L >= self.region_size:
            x = x.view(B, -1, self.region_size, D)

        x = x.split(
            D // self.k,
            dim=-1
        )

        res = []
        for idx, chunk in enumerate(x):
            chunk = torch.roll(chunk, idx * self.scale, dims=-2)
            res.append(chunk)

        res = torch.cat(res, dim=-1)

        if L >= self.region_size:
            res = res.view(B, L, D)

        return res


# net = IntraRegionK(512, 64)

# inputs = torch.randn(1, 100, 512)

# out = net(inputs)

# net = IntraRegionK(513, 64)

# inputs = torch.randn(1, 100, 512)

# out = net(inputs)

# k = 4
# dim = 8
# len = 33

# torch.set_printoptions(linewidth=200)
# net = IntraRegionK(dim,  k, k ** 3)

# a = [torch.arange(len) for i in range(dim)]
# a = torch.stack(a, dim=-1).unsqueeze(dim=0)
# inputs = a
# out = net(inputs)
# print(net.region_size)
# print(net.k)
# print(net.scale)
