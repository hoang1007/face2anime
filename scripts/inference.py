import torch
from argparse import ArgumentParser
from PIL import Image
from torchvision.utils import save_image

from face2anime.model import CycleGAN

import torchvision.transforms as T

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='checkpoints/epoch=199-step=498400.ckpt')
    parser.add_argument('--image_path', type=str, default='time-city.jpg')
    args = parser.parse_args()
    return args


def main(args):
    image_size = (256, 256)
    transform=T.Compose((
        T.Resize(image_size),
        T.ToTensor(),
        T.Normalize([0.5], [0.5])
    ))

    image = Image.open(args.image_path).convert('RGB')
    image = transform(image).to('cuda')

    model = CycleGAN.load_from_checkpoint(args.checkpoint, strict=False)
    model.eval()
    model.to('cuda')

    pred = model(image)
    pred = (pred.cpu() * 0.5) + 0.5
    save_image(pred, 'output.jpg')

if __name__ == '__main__':
    main(parse_args())
