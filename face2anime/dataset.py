from typing import Callable, Optional, List
import os
import random

from PIL import Image
import torch
from torch.utils.data import Dataset


class CycleGANDataset(Dataset):
    def __init__(
        self,
        root: str,
        prefix_a: str = 'A',
        prefix_b: str = 'B',
        transform: Optional[Callable] = None,
        aligned: bool = False
    ):
        self.transform = transform
        self.aligned = aligned

        self.__impaths_a = self.list_imgs(os.path.join(root, prefix_a))
        self.__impaths_b = self.list_imgs(os.path.join(root, prefix_b))

    def list_imgs(self, dir: str):
        im_exts = ('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')
        impaths: List[str] = []
        for f in os.scandir(dir):
            if os.path.splitext(f.name)[1] in im_exts:
                impaths.append(f.path)
        return impaths
    
    def __len__(self):
        return max(len(self.__impaths_a), len(self.__impaths_b))
    
    def __getitem__(self, idx: int):
        impath_a = self.__impaths_a[idx % len(self.__impaths_a)]

        if self.aligned:
            impath_b = self.__impaths_b[idx % len(self.__impaths_b)]
        else:
            rand_idx = torch.randint(0, len(self.__impaths_b), size=(1,)).item()
            impath_b = self.__impaths_b[rand_idx]
        
        im_a = Image.open(impath_a).convert('RGB')
        im_b = Image.open(impath_b).convert('RGB')

        if self.transform is not None:
            im_a = self.transform(im_a)
            im_b = self.transform(im_b)
        
        return im_a, im_b
