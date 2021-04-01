from ..config import *
from typing import Callable, Union, Sequence, Dict
import torch
from torch import nn

from .attn_blocks import MHSA


class BasicConvolutionalBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            dilation: int = 1,
            n_layers: int = 2,
            normalize: bool = False,
            use_mhsa: bool = False,
            mhsa_heads: int = 4,
            activation: Callable = nn.ReLU,
            downsample: nn.Module = nn.Identity()):
        super(BasicConvolutionalBlock, self).__init__()
        assert n_layers >= 1
        padding = (dilation * (kernel_size - 1.)) / 2.
        if padding == int(padding):
            padding = int(padding)
        else:
            raise ValueError(f'Padding must be an integer, but was {padding:0.2f}')
        if downsample is None:
            downsample = nn.Identity()

        layers = [nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            padding_mode='circular'
        )]
        if normalize:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(activation(inplace=True))
        if use_mhsa:
            layers.append(MHSA(
                in_channels=out_channels,
                heads=mhsa_heads,
                curr_h=N_ROWS,
                curr_w=N_COLS
            ))

        for i in range(n_layers - 1):
            layers.append(nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                padding_mode='circular'
            ))
            if normalize:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(activation(inplace=True))
        # Remove final activation layer - to be applied after residual connection
        layers = layers[:-1]
        layers.append(downsample)
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class SELayer(nn.Module):
    def __init__(self, n_channels, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(n_channels, n_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(n_channels // reduction, n_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ResidualModel(nn.Module):
    def __init__(
            self,
            block_class: Union[BasicConvolutionalBlock],
            conv_block_kwargs: Sequence[Dict],
            squeeze_excitation: bool = True,
    ):
        super(ResidualModel, self).__init__()
        assert len(conv_block_kwargs) >= 1
        self.conv_blocks = nn.ModuleList()
        self.change_n_channels = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        self.se_layers = nn.ModuleList()
        self.activations = nn.ModuleList()
        for kwargs in conv_block_kwargs:
            self.conv_blocks.append(block_class(**kwargs))
            if kwargs['in_channels'] != kwargs['out_channels']:
                self.change_n_channels.append(nn.Conv2d(kwargs['in_channels'], kwargs['out_channels'], 1))
            else:
                self.change_n_channels.append(nn.Identity())
            self.downsamplers.append(kwargs.get('downsample', nn.Identity()))
            self.se_layers.append(SELayer(kwargs['out_channels']) if squeeze_excitation else nn.Identity())
            self.activations.append(kwargs.get('activation', nn.ReLU)(inplace=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for cb, cnc, d, se, a in zip(
            self.conv_blocks,
            self.change_n_channels,
            self.downsamplers,
            self.se_layers,
            self.activations
        ):
            identity = x
            x = se(cb(x))
            x = x + d(cnc(identity))
            x = a(x)
        return x
