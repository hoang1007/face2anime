from typing import Union
import torch
from torch import nn


def gan_loss(
    input: torch.Tensor,
    target: Union[int, bool, torch.Tensor],
    lsgan: bool = True,
    reduction: str = "mean",
):
    if not isinstance(target, torch.Tensor):
        target = torch.tensor(target).type_as(input).expand_as(input)

    if lsgan:
        return nn.functional.mse_loss(
            input, target, reduction=reduction
        )
    else:
        return nn.functional.binary_cross_entropy_with_logits(
            input, target, reduction=reduction
        )
