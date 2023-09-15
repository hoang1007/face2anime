import torch
from torch import nn

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
            nn.InstanceNorm2d(in_channels)
        )
    
    def forward(self, x: torch.Tensor):
        return x + self.blocks(x)

