import copy

import torch

import models.configs as configs

from os.path import join as pjoin

from torch import nn
from torch.nn.modules.utils import _pair

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


class MlpMixer(nn.Module):
    def __init__(self, input_dim, hidden_dim, seq_len, num_tokens, channel_dim, token_dim, num_classes=2, ):
        super(MlpMixer, self).__init__()
        # self.zero_head = zero_head
        self.num_classes = num_classes
        self.seq_len = seq_len
        self.token_dim = token_dim

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        # self.stem = nn.Conv2d(in_channels=3,
        #                     #   out_channels=config.hidden_dim,
        #                       kernel_size=patch_size,
        #                       stride=patch_size)
        self.head = nn.Linear(hidden_dim, num_classes, bias=True)
        self.pre_head_ln = nn.LayerNorm(hidden_dim, eps=1e-6)


        # self.layer = nn.ModuleList()
        # for _ in range(config.num_blocks):
        # for _ in range(2):
        # seq_len =
        self.layer1 = ReducedMixerBlock(
            wsi_len=seq_len,
            num_tokens=num_tokens,
            channel_dim=channel_dim,
            hidden_dim=hidden_dim
        )
        self.layer2 = MixerBlock(
            hidden_dim=hidden_dim,
            num_tokens=num_tokens,
            token_dim=token_dim,
            channel_dim=channel_dim
        )

        # self.layer.append(copy.deepcopy(layer))

     # 44380
    def forward(self, x, labels=None):
        # x = self.stem(x)
        # x = x.flatten(2)
        # x = x.transpose(-1, -2)

        # for block in self.layer:
            # x = block(x)
        x = self.fc1(x)


        x = self.layer1(x)
        x = self.layer2(x)
        x = self.pre_head_ln(x)
        x = torch.mean(x, dim=1)
        logits = self.head(x)

        # print(logits)
        # print(logits.shape, logits.view(-1, self.num_classes).shape, labels.view(-1).shape)
        # torch.Size([128, 3, 224, 224]) torch.Size([128])
        # print(logits.shape)
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))
            return loss
        else:
            return logits

CONFIGS = {
    'Mixer-B_16': configs.get_mixer_b16_config(),
    'Mixer-L_16': configs.get_mixer_l16_config(),
    'Mixer-B_16-21k': configs.get_mixer_b16_config(),
    'Mixer-L_16-21k': configs.get_mixer_l16_config()
}
