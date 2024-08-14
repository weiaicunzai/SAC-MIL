import math

import torch

from torch import nn
import torch.nn.functional as F

from models.Pos import RotaryEmbedding


class MlpMixer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes=2):
        super().__init__()
        self.num_classes = num_classes

        self.fc1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )

        self.k = 64

        self.intra_region = nn.Sequential(
            IntraRegionK(hidden_dim, self.k, self.k ** 1),
            IntraRegionK(hidden_dim, self.k, self.k ** 2),
            IntraRegionK(hidden_dim, self.k, self.k ** 3)
        )

        self.fc2 = nn.Linear(hidden_dim, self.num_classes)

        self.pos_emb = RotaryEmbedding(hidden_dim, scale=1)

    def forward(self, **kwargs):

        x = kwargs['data'].float() #[B, n, 1024]

        pos = kwargs['pos']

        x = self.fc1(x)

        x = self.pos_emb(x, pos)


        x = self.intra_region(x)

        x = x.mean(dim=-2)

        logits = self.fc2(x)

        Y_prob = F.softmax(logits, dim = 1)
        results_dict = {'logits': logits, 'Y_prob': Y_prob}

        return results_dict

class IntraRegionK(nn.Module):
    def __init__(self, hidden_dim, k, region_size):
        super().__init__()

        self.k = k
        self.region_size = region_size
        self.scale = int(self.k ** (math.log(self.region_size, self.k) - 1))

        self.inter_fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.intra_fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.pre_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):

        res = x
        x = self.pre_norm(x)
        x = self.roll_all(x)
        x = self.inter_fc(x)
        x = self.unroll_all(x)
        x = self.intra_fc(x)

        x = x + res

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
