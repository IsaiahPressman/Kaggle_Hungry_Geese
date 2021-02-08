import torch
from torch import distributions, nn
import torch.nn.functional as F
from typing import *


class ConvolutionalBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            dilation: int = 1,
            n_layers: int = 2,
            activation_func: nn.Module = nn.ReLU(),
            downsample: nn.Module = nn.Identity()):
        super(ConvolutionalBlock, self).__init__()
        assert n_layers >= 1
        padding = (dilation * (kernel_size - 1.)) / 2.
        if padding == int(padding):
            padding = int(padding)
        else:
            raise ValueError(f'Padding must be an integer, but was {padding:0.2f}')
        if downsample is None:
            downsample = nn.Identity()

        layers = [
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
                padding_mode='circular'
            ),
            nn.BatchNorm2d(out_channels),
            activation_func
        ]
        for i in range(n_layers - 1):
            layers.append(nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
                padding_mode='circular'
            ))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(activation_func)
        # Remove final activation layer - to be applied after residual connection
        layers = layers[:-1]
        layers.append(downsample)
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class ResidualNetwork(nn.Module):
    def __init__(self, conv_block_kwargs: Sequence[Dict]):
        super(ResidualNetwork, self).__init__()
        assert len(conv_block_kwargs) >= 1
        self.conv_blocks = nn.ModuleList()
        self.change_n_channels = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        self.activation_funcs = nn.ModuleList()
        for kwargs in conv_block_kwargs:
            self.conv_blocks.append(ConvolutionalBlock(**kwargs))
            if kwargs['in_channels'] != kwargs['out_channels']:
                self.change_n_channels.append(nn.Conv2d(kwargs['in_channels'], kwargs['out_channels'], 1))
            else:
                self.change_n_channels.append(nn.Identity())
            self.downsamplers.append(kwargs.get('downsample', nn.Identity()))
            self.activation_funcs.append(kwargs.get('activation_func', nn.ReLU()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for cb, cnc, d, a in zip(
                self.conv_blocks,
                self.change_n_channels,
                self.downsamplers,
                self.activation_funcs
        ):
            identity = x
            x = cb(x)
            x = x + d(cnc(identity))
            x = a(x)
        return x


class BasicActorCriticNetwork(nn.Module):
    def __init__(
            self,
            conv_block_kwargs: Sequence[Dict],
            fc_in_channels: int
    ):
        super(BasicActorCriticNetwork, self).__init__()
        self.base = ResidualNetwork(conv_block_kwargs)
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # fc_in_channels = conv_block_kwargs[-1]['out_channels']
        self.actor = nn.Linear(fc_in_channels, 4)
        self.critic = nn.Linear(fc_in_channels, 1)

    def forward(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        base_out = self.base(states)
        base_out = base_out.view(base_out.shape[0], -1)
        return self.actor(base_out), self.critic(base_out).squeeze(dim=-1)

    def sample_action(self, states: torch.Tensor, train: bool = False):
        if train:
            logits, values = self.forward(states)
        else:
            with torch.no_grad():
                logits, values = self.forward(states)
        probs = F.softmax(logits, dim=-1)
        m = distributions.Categorical(probs)
        sampled_actions = m.sample()
        if train:
            return sampled_actions, (logits, values)
        else:
            return sampled_actions

    def choose_best_action(self, states: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            logits, _ = self.forward(states)
            return logits.argmax(dim=-1)
