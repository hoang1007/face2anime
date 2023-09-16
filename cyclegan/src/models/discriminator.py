from torch import nn
from pytorch_lightning import LightningModule


def get_discriminator_block(input_dim, output_dim):
    return nn.Sequential(nn.Linear(input_dim, output_dim), nn.LeakyReLU(negative_slope=0.2))


class Discriminator(LightningModule):
    def __init__(self, im_dim=784, hidden_dim=128):
        super().__init__()
        self.disc = nn.Sequential(
            get_discriminator_block(im_dim, hidden_dim * 4),
            get_discriminator_block(hidden_dim * 4, hidden_dim * 2),
            get_discriminator_block(hidden_dim * 2, hidden_dim),
            nn.Linear(hidden_dim, 1),
        )
        
    def forward(self, image):
        return self.disc(image)

    # def on_backward(self, use_amp, loss, optimizer):
    #     loss.backward(retain_graph=True)

if __name__ == "__main__":
    _ = Discriminator()