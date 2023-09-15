from typing import Dict

from math import sqrt
import torch
from torchvision.utils import make_grid


def make_image_grid(img_dict: Dict[str, torch.Tensor]):
    out = dict()
    for title, imgs in img_dict.items():
        if imgs.ndim == 3:
            imgs = imgs.unsqueeze(0)
        
        n_imgs = imgs.size(0)
        n_row = int(sqrt(n_imgs))
        im_grid = make_grid(imgs, nrow=n_row)

        out[title] = im_grid
    
    return out
