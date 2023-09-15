from typing import List
import torch
from torch import nn


class ResnetGenerator(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: List[int] = [64, 128],
        num_res_layers: int = 6,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.in_proj = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, hidden_channels[0], kernel_size=7),
            nn.InstanceNorm2d(hidden_channels[0]),
            nn.ReLU(inplace=True),
        )

        self.downsample = nn.ModuleList()
        prev_channels = hidden_channels[0]
        for channels in hidden_channels:
            self.downsample.append(
                nn.Sequential(
                    nn.Conv2d(
                        prev_channels, channels, kernel_size=3, stride=2, padding=1
                    ),
                    nn.InstanceNorm2d(channels),
                    nn.ReLU(inplace=True),
                )
            )

        self.res_blocks = nn.ModuleList()
        for _ in range(num_res_layers):
            self.res_blocks.append(ResBlock(prev_channels, dropout=dropout))

        self.upsample = nn.ModuleList()
        for channels in reversed(hidden_channels):
            self.upsample.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        channels,
                        prev_channels,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1,
                    ),
                    nn.InstanceNorm2d(prev_channels),
                    nn.ReLU(inplace=True),
                )
            )
            prev_channels = channels

        self.out_proj = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(hidden_channels[0], out_channels, kernel_size=7),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor):
        x = self.in_proj(x)

        for block in self.downsample:
            x = block(x)
        for block in self.res_blocks:
            x = block(x)
        for block in self.upsample:
            x = block(x)

        out = self.out_proj(x)
        return out


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, dropout: float = 0.0, bias: bool = True):
        self.blocks = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, bias=bias),
            nn.InstanceNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, bias=bias),
            nn.InstanceNorm2d(in_channels),
        )

    def forward(self, x: torch.Tensor):
        return x + self.blocks(x)
