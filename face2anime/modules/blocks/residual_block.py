from typing import Optional

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """
    ### Residual block of resnet backbone
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: Optional[int] = None,
                 drop_rate: float = 0.):
        """
        in_channels: number of input channels
        out_channels: number of out channels. defaults to `in_channels.
        drop_rate: parameter of dropout layer
        """
        super().__init__()

        # `out_channels` not specified
        if out_channels is None:
            out_channels = in_channels

        # Normalization and convolution in input layer
        self.in_layers = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=in_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=3,
                      padding=1),
        )

        # Normalization and convolution in output layers
        self.out_layers = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=out_channels),
            nn.SiLU(inplace=True),
            nn.Dropout(p=drop_rate),
            nn.Conv2d(in_channels=out_channels,
                      out_channels=out_channels,
                      kernel_size=3,
                      padding=1),
        )

        # `channels` to `out_channels` mapping layer for residual connection
        if out_channels == in_channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Conv2d(in_channels=in_channels,
                                             out_channels=out_channels,
                                             kernel_size=1)

    def forward(self,
                x: torch.Tensor,
                t_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: is the input feature map with shape `[batch_size, channels, height, width]`
        t_emb: is the time step embeddings of shape `[batch_size, d_t_emb]`. defaults to None
        """

        # Normalization and convolution in input layers
        out = self.in_layers(x)

        if t_emb is not None:
            # Time step embeddings
            t_emb = self.emb_layers(t_emb).type(out.dtype)
            # Add time step embeddings
            out = out + t_emb[:, :, None, None]

        # Normalization and convolution in output layers
        out = self.out_layers(out)

        # Add skip connection
        return self.skip_connection(x) + out


if __name__ == "__main__":
    x = torch.randn(2, 32, 10, 10)
    resnetBlock = ResidualBlock(
        in_channels=32,
        out_channels=64,
    )
    out = resnetBlock(x)
    print('Input:', x.shape)
    print('Output:', out.shape)