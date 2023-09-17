from argparse import ArgumentParser

from face2anime.model import CycleGAN, CycleGANTrainingConfig
from face2anime.modules.generators import init_generator
from face2anime.modules.discriminators import init_discriminator
from face2anime.dataset import CycleGANDataset

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers.wandb import WandbLogger
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as T


if __name__ == '__main__':
    batch_size = 32
    epochs = 200

    training_cfg = CycleGANTrainingConfig(
        learning_rate=1e-5,
        weight_decay=1e-4,
        use_lsgan=True,
        warmup_generator_steps=1000
    )

    Generator = init_generator(
        'base',
        img_channels=3,
        n_layer_blocks=2,
    )
    Discriminator = init_discriminator(
        'nlayer',
        ndf=64
    )
    model = CycleGAN(
        generator_ab=Generator(),
        discriminator_a=Discriminator(),
        generator_ba=Generator(),
        discriminator_b=Discriminator(),
        training_config=training_cfg
    )

    dataset = CycleGANDataset(
        'data',
        prefix_a='face_photo',
        prefix_b='face_cartoon/kyoto_face',
        transform=T.Compose((
            T.Resize(64),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize([0.5], [0.5])
        ))
    )
    train_dts, val_dts = random_split(dataset, [0.8, 0.2])
    train_dataloader = DataLoader(
        train_dts,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )
    val_dataloader = DataLoader(
        val_dts,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )

    trainer = Trainer(
        callbacks=[
            ModelCheckpoint('checkpoints'),
        ],
        logger=WandbLogger(project='face2anime', log_model=True),
        precision='16-mixed',
        max_epochs=epochs,
    )

    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
