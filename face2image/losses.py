from typing import Union
import torch
from torch import nn


def gan_loss(
    input: torch.Tensor,
    target: Union[int, torch.Tensor],
    lsgan: bool = True,
    size_average: bool = False,
    reduction: str = "mean",
):
    if not isinstance(target, torch.Tensor):
        target = torch.tensor(target).expand_as(input)

    if lsgan:
        return nn.functional.mse_loss(
            input, target, size_average=size_average, reduction=reduction
        )
    else:
        return nn.functional.binary_cross_entropy(
            input, target, size_average=size_average, reduction=reduction
        )
