from typing import List
import torch
from torch import nn
from torch import Tensor

from face2anime.modules.up_down import Encoder, Decoder


class BaseGenerator(nn.Module):
    def __init__(self,
                 img_channels: int,
                 channels: int = 32,
                 block: str = "Residual",
                 n_layer_blocks: int = 1,
                 channel_multipliers: List[int] = [1, 2, 4],
                 attention: str = "SelfAttention"):
        
        super().__init__()
        
        self.encoder = Encoder(in_channels=img_channels,
                               channels=channels,
                               block=block,
                               n_layer_blocks=n_layer_blocks,
                               channel_multipliers=channel_multipliers,
                               attention=attention)
        
        self.decoder = Decoder(out_channels=img_channels,
                               channels=channels,
                               block=block,
                               n_layer_blocks=n_layer_blocks,
                               channel_multipliers=channel_multipliers,
                               attention=attention)

    def forward(self, x: Tensor):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    

if __name__ == "__main__":
    x = torch.randn(2, 3, 32, 32)
    generator = BaseGenerator(img_channels=3)
    out = generator(x)

    print('***** Generator *****')
    print('Input:', x.shape)
    print('Output:', out.shape)