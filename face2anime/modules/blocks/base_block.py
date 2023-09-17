from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseBlock(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: Optional[int] = None,
                 drop_rate: float = 0.,
                 residual: bool = False) -> None:
        """
        in_channels: the number of input channels
        out_channels: is the number of out channels. defaults to `in_channels.
        d_t_emb: the size of timestep embeddings if not None. defaults to None
        """
        super().__init__()
        self.residual = residual

        # `out_channels` not specified
        if out_channels is None:
            out_channels = in_channels

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=3,
                      padding=1,
                      bias=False),
            nn.GroupNorm(num_groups=32, num_channels=out_channels),
            nn.SiLU(inplace=True),
            nn.Dropout(p=drop_rate),
            nn.Conv2d(in_channels=out_channels,
                      out_channels=out_channels,
                      kernel_size=3,
                      padding=1,
                      bias=False),
            nn.GroupNorm(num_groups=32, num_channels=out_channels),
        )

    def forward(self,
                x: torch.Tensor) -> torch.Tensor:
        """
        x: is the input feature map with shape `[batch_size, channels, height, width]`
        """
        # add skip connection
        if self.residual:
            out = F.silu(x + self.double_conv(x))
        else:
            out = self.double_conv(x)

        return out


if __name__ == "__main__":
    x = torch.randn(2, 32, 10, 10)
    t = torch.randn(2, 32)
    baseBlock = BaseBlock(
        in_channels=32,
        out_channels=64
    )
    out = baseBlock(x)
    print('***** BaseBlock *****')
    print('Input:', x.shape)
    print('Output:', out.shape)