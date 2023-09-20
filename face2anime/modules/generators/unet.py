from typing import List, Optional

import torch
import torch.nn as nn
from torch import Tensor

from face2anime.modules.blocks import init_block
from face2anime.modules.attentions import init_attention
from face2anime.modules.up_down import DownSample, UpSample

class UNetGenerator(nn.Module):
    """
    ### Unet model
    """

    def __init__(self,
                 img_channels: int,
                 channels: int = 64,
                 block: str = "Residual",
                 n_layer_blocks: int = 1,
                 channel_multipliers: List[int] = [1, 2, 4, 4],
                 attention: str = "SelfAttention") -> None:
        """
        img_channels: the number of channels in the input feature map
        channels: the base channel count for the model
        block: name of block for each level
        n_layer_blocks: number of blocks at each level
        channel_multipliers: the multiplicative factors for number of channels for each level
        attention: name of attentions for each level
        """
        super().__init__()

        self.base_channels = channels

        # number of levels (downSample and upSample)
        levels = len(channel_multipliers)

        # number of channels at each level
        channels_list = [channels * m for m in channel_multipliers]

        # block to downSample
        Block = init_block(block)

        # attention layer
        Attention = init_attention(attention) if attention is not None else None

        # input half of the U-Net
        self.down = nn.ModuleList()

        # input convolution
        self.down.append(nn.Conv2d(img_channels, channels, 3, padding=1))

        # number of channels at each block in the input half of U-Net
        input_block_channels = [channels]

        # prepare for input half of U-net
        for i in range(levels):
            # add the blocks, attentions
            for _ in range(n_layer_blocks):
                layers = [
                    Block(
                        in_channels=channels,
                        out_channels=channels_list[i]
                    )
                ]

                channels = channels_list[i]
                input_block_channels.append(channels)

                self.down.append(nn.Sequential(*layers))

            # down sample at all levels except last
            if i != levels - 1:
                self.down.append(DownSample(channels=channels))
                input_block_channels.append(channels)

        # the middle of the U-Net
        self.mid = nn.Sequential(
            Block(in_channels=channels,),
            Attention(channels=channels) if Attention is not None else Block(channels, channels),
            Block(in_channels=channels,),
        )

        # second half of the U-Net
        self.up = nn.ModuleList([])

        # prepare layer for upSampling
        for i in reversed(range(levels)):
            # add the blocks, attentions


            for j in range(n_layer_blocks + 1):
                layers = [
                    Block(
                        in_channels=channels + input_block_channels.pop(),
                        out_channels=channels_list[i],
                    )
                ]
                channels = channels_list[i]

                if i != 0 and j == n_layer_blocks:
                    layers.append(UpSample(channels))

                self.up.append(nn.Sequential(*layers))

        self.conv_out = nn.Sequential(
            nn.GroupNorm(32, channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, img_channels, 3, padding=1),
        )

    def forward(self, x: Tensor):
        """
        :param x: is the input feature map of shape `[batch_size, channels, width, height]`
        """

        # to store the input half outputs for skip connections
        x_input_block = []

        # input half of the U-Net
        for module in self.down:
            x = module(x)
            x_input_block.append(x)

        # middle of the U-Net
        x = self.mid(x)

        # Output half of the U-Net
        for module in self.up:
            x = torch.cat([x, x_input_block.pop()], dim=1)
            x = module(x)

        # output convolution
        x = self.conv_out(x)

        #
        return x


if __name__ == "__main__":
    unet = UNetGenerator(img_channels=3)
    x = torch.randn(2, 3, 64, 64)
    out = unet(x)

    print('***** UNetGenerator *****')
    print('Input:', x.shape)
    print('Output:', out.shape)