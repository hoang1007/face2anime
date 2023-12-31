from typing import List

import torch
from torch import nn
from torch import Tensor


from face2anime.modules.blocks import init_block
from face2anime.modules.attentions import init_attention
from face2anime.modules.up_down import DownSample


class BaseDiscriminator(nn.Module):
    def __init__(self,
                 img_channels: int,
                 out_channels: int,
                 channels: int = 32,
                 block: str = "Residual",
                 n_layer_blocks: int = 1,
                 channel_multipliers: List[int] = [1, 2, 4],
                 attention: str = "Attention",
                 ):
        
        super(BaseDiscriminator, self).__init__()

        # Number of levels downSample
        levels = len(channel_multipliers)

        # Number of channels at each level
        channels_list = [channels * m for m in channel_multipliers]

        # Block to downSample
        Block = init_block(block)

        # attention layer
        Attention = init_attention(attention) if attention is not None else None

        # Input convolution
        self.layer_input = nn.Conv2d(in_channels=img_channels,
                                     out_channels=channels,
                                     kernel_size=3,
                                     padding=1)

        self.layers = nn.ModuleList()

        # Prepare layer for downSampling
        for i in range(levels):
            # Add the blocks and downSample
            blocks = nn.ModuleList()

            for _ in range(n_layer_blocks):
                blocks.append(
                    Block(
                        in_channels=channels,
                        out_channels=channels_list[i],
                    ))

                channels = channels_list[i]

            down = nn.Module()
            down.blocks = blocks

            # Down-sampling at the end of each top level block except the last
            if i != levels - 1:
                down.downSample = DownSample(channels=channels)
            else:
                down.downSample = nn.Identity()

            #
            self.layers.append(down)
        
        # classifier block
        self.classifier = nn.Sequential(
            Block(in_channels=channels),
            Attention(channels=channels) if Attention is not None else Block(channels, channels),
            nn.Conv2d(channels, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x: Tensor):
        x = self.layer_input(x)
        
        for layer in self.layers:
            # Blocks
            for block in layer.blocks:
                x = block(x)
            # Down-sampling
            x = layer.downSample(x)
        
        x = self.classifier(x)

        return x
    

if __name__ == "__main__":
    x = torch.randn(2, 3, 32, 32)
    discriminator = BaseDiscriminator(img_channels=3,
                                      out_channels=1,
                                      attention=None)
    out = discriminator(x)

    print('***** Discriminator *****')
    print('Input:', x.shape)
    print('Output:', out.shape)