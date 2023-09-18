from argparse import ArgumentParser
from omegaconf import OmegaConf, DictConfig

from face2anime.utils import init_module
from face2anime.model import CycleGAN, CycleGANTrainingConfig
from face2anime.dataset import CycleGANDataset

from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers.wandb import WandbLogger
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as T


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    args = parser.parse_args()
    return args


def load_config(path):
    config = OmegaConf.load(path)
    return config


def main(config: DictConfig):
    log_config = config.copy()

    seed = config.get('training').pop('seed')
    seed_everything(seed)

    batch_size = config.get('training').pop('batch_size', 1)
    epochs = config.get('training').pop('epochs', 100)
    lr = config.get('training').pop('learning_rate', 5e-5)
    weight_decay = config.get('training').pop('weight_decay', 0.0)
    use_lsgan = config.get('training').pop('use_lsgan', True)
    warmup_generator_steps = config.get('training').pop('warmup_generator_steps', 0)

    training_cfg = CycleGANTrainingConfig(
        learning_rate=lr,
        weight_decay=weight_decay,
        use_lsgan=use_lsgan,
        warmup_generator_steps=warmup_generator_steps
    )

    model = CycleGAN(
        generator_ab=init_module(config.generator),
        discriminator_a=init_module(config.discriminator),
        generator_ba=init_module(config.generator),
        discriminator_b=init_module(config.discriminator),
        training_config=training_cfg
    )

    dataset = CycleGANDataset(
        config.data.root,
        prefix_a=config.data.prefix_a,
        prefix_b=config.data.prefix_b,
        transform=T.Compose((
            T.Resize(config.data.image_size),
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

    logger = WandbLogger(project='face2anime', log_model=True)
    logger.log_hyperparams(log_config)

    trainer = Trainer(
        callbacks=[
            ModelCheckpoint('checkpoints'),
        ],
        logger=logger,
        max_epochs=epochs,
        **config.training
    )

    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)


if __name__ == '__main__':
    args = parse_args()
    config = load_config(args.config)
    main(config)
