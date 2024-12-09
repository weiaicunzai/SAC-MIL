import math

import torch

from torch import nn
import torch.nn.functional as F

from models.pos import RotaryEmbedding


class MlpMixer(nn.Module):
    def __init__(self, input_dim, hidden_dim, scale, num_classes=2):
        super().__init__()
        self.num_classes = num_classes

        self.fc1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )

        self.k = 64

        self.intra_region1 = IntraRegionK(hidden_dim, self.k, self.k ** 1, use_pos=True, scale=scale) 
        self.intra_region2 = IntraRegionK(hidden_dim, self.k, self.k ** 2, use_pos=False, scale=scale) 
        self.intra_region3 = IntraRegionK(hidden_dim, self.k, self.k ** 3, use_pos=False, scale=scale) 

        self.fc2 = nn.Linear(hidden_dim, self.num_classes)

    def forward(self, **kwargs):

        x = kwargs['data'].float() #[B, n, 1024]

        pos = kwargs['pos']

        x = self.fc1(x)

        x = self.intra_region1(x, pos)
        x = self.intra_region2(x, pos)
        x = self.intra_region3(x, pos)

        x = x.mean(dim=-2)

        logits = self.fc2(x)

        Y_prob = F.softmax(logits, dim = 1)
        results_dict = {'logits': logits, 'Y_prob': Y_prob}

        return results_dict

class IntraRegionK(nn.Module):
    def __init__(self, hidden_dim, k, region_size, use_pos, scale):
        super().__init__()

        self.k = k
        self.region_size = region_size
        layer_idx = math.log(self.region_size, self.k) - 1
        self.scale = int(self.k ** layer_idx)
        # self.scale = int(self.k ** (math.log(self.region_size, self.k) - 1))
        # assert self.scale == int(self.k ** (math.log(self.region_size, self.k) - 1))

        self.inter_fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        if use_pos:
            self.intra_fc = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, 1),
                )
        else:
            self.intra_fc = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
            )

        print(self.intra_fc)

        self.norm1 = nn.LayerNorm(hidden_dim)

        self.use_pos = use_pos
        print('use_pos: ', self.use_pos)
        if self.use_pos is True:
            from models.pos import RotaryEmbedding as pos_emb
            self.pos_emb = pos_emb(hidden_dim, scale=scale)

        self.hidden_dim = hidden_dim


    def mixer_attn(self, x, pos):
        res = x
        x = self.norm1(x)

        if self.use_pos:
            # x = self.pos_emb.rotate_queries_or_keys(x, pos)
            x = self.pos_emb(x, pos)

        x = self.roll_all(x)
        x = self.inter_fc(x)
        x = self.unroll_all(x)
        x = self.intra_fc(x)

        if self.use_pos:
            # x = x.softmax(dim=1)
            # 87+
            x = (x * res).sum(dim=1, keepdim=True)

        x = res + x
        return x


    def forward(self, x, pos):


        x = self.mixer_attn(x, pos)


        return x

    def split_parts(self, n):
        if n < self.region_size:
            return n

        mod = n % self.region_size
        num_part1 = n - mod

        assert num_part1 > 0

        return num_part1

    def roll_all(self, x):
        num_tokens = x.shape[1]

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
