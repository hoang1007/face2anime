from typing import Dict, Optional, Any
import itertools

from lightning.pytorch import LightningModule
from lightning.pytorch.utilities.types import STEP_OUTPUT
import torch
from torch import nn
import torch.nn.functional as F

from torchmetrics.image import FrechetInceptionDistance
from face2anime.utils import ImagePool

from .losses import gan_loss


class CycleGANTrainingConfig:
    def __init__(
        self,
        lambda_ab: float = 10.0,
        lambda_ba: float = 10.0,
        identity_loss_weight: float = 0.5,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        use_lsgan: bool = True,
        pool_size: int = 0,
        warmup_generator_steps: int = 0
    ):
        self.lambda_ab = lambda_ab
        self.lambda_ba = lambda_ba
        self.identity_loss_weight = identity_loss_weight
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.use_lsgan = use_lsgan
        self.pool_size = pool_size
        self.warmup_generator_steps = warmup_generator_steps
        self.feature = 64
        self.n_image_fid = 100


class CycleGAN(LightningModule):
    def __init__(
        self,
        generator_ab: nn.Module,
        generator_ba: nn.Module,
        discriminator_a: Optional[nn.Module] = None,
        discriminator_b: Optional[nn.Module] = None,
        training_config: Optional[CycleGANTrainingConfig] = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.automatic_optimization = False
        self.training_config = training_config

        self.fake_a_pool = ImagePool(training_config.pool_size)
        self.fake_b_pool = ImagePool(training_config.pool_size)

        self.generator_ab = generator_ab
        self.discriminator_b = discriminator_b

        self.generator_ba = generator_ba
        self.discriminator_a = discriminator_a

        self.fid_photo = FrechetInceptionDistance(feature=training_config.feature, 
                                                                    normalize=True)
        
        self.fid_anime = FrechetInceptionDistance(feature=training_config.feature, 
                                                                    normalize=True)
        
        self.n_image_fid = training_config.n_image_fid
        self.photo_images = {
            'train': None,
            'val': None,
            'test': None,
        }
        self.anime_images = {
            'train': None,
            'val': None,
            'test': None,
        }

    def forward(self, input: torch.FloatTensor, b2a: bool = False):
        if b2a:
            return self.generator_ba(input)
        else:
            return self.generator_ab(input)

    def compute_generator_losses(
        self,
        real_a: torch.FloatTensor,
        fake_a: torch.FloatTensor,
        real_b: torch.FloatTensor,
        fake_b: torch.FloatTensor,
    ):
        assert self.training_config is not None, 'Training config is required for training state!'
        real_target = 1
        identity_loss_weight = self.training_config.identity_loss_weight
        lambda_ab = self.training_config.lambda_ab
        lambda_ba = self.training_config.lambda_ba
        lsgan = self.training_config.use_lsgan

        # Loss for generator A
        losses_a = {
            "identity": F.l1_loss(self.generator_ab(real_b), real_b) * identity_loss_weight * lambda_ab,
            "gan": gan_loss(self.discriminator_b(fake_b), real_target, lsgan),
            "cycle": F.l1_loss(self.generator_ab(fake_a), real_b) * lambda_ab,
        }

        losses_b = {
            "identity": F.l1_loss(self.generator_ba(real_a), real_a)  * identity_loss_weight * lambda_ba,
            "gan": gan_loss(self.discriminator_a(fake_a), real_target, lsgan),
            "cycle": F.l1_loss(self.generator_ba(fake_b), real_a) * lambda_ba,
        }

        return losses_a, losses_b

    def compute_discriminator_losses(
        self,
        real_a: torch.FloatTensor,
        fake_a: torch.FloatTensor,
        real_b: torch.FloatTensor,
        fake_b: torch.FloatTensor,
    ):
        assert self.training_config is not None, 'Training config is required for training state!'

        fake_target, real_target = 0, 1
        lsgan = self.training_config.use_lsgan

        fake_a = self.fake_a_pool.query(fake_a)
        fake_b = self.fake_b_pool.query(fake_b)

        discriminator_loss_a = 0.5 * (
            gan_loss(self.discriminator_a(fake_a), fake_target, lsgan) + gan_loss(self.discriminator_a(real_a), real_target, lsgan)
        )
        discriminator_loss_b = 0.5 * (
            gan_loss(self.discriminator_b(fake_b), fake_target, lsgan) + gan_loss(self.discriminator_b(real_b), real_target, lsgan)
        )

        losses_a = {
            'discriminator': discriminator_loss_a,
        }
        losses_b = {
            'discriminator': discriminator_loss_b
        }

        return losses_a, losses_b

    def training_step(self, batch, batch_idx: int):
        real_a, real_b = batch

        opt_g, opt_d = self.optimizers()
        loss_dict = dict()
        prefix = 'train/'

        # Train generators
        self.toggle_optimizer(opt_g)
        opt_g.zero_grad()
        fake_b = self(real_a)
        fake_a = self(real_b, b2a=True)
        g_losses_a, g_losses_b = self.compute_generator_losses(
            real_a, fake_a, real_b, fake_b
        )
        g_loss = sum(g_losses_a.values()) + sum(g_losses_b.values())
        self.manual_backward(g_loss)
        opt_g.step()
        self.untoggle_optimizer(opt_g)

        for n, loss in g_losses_a.items():
            loss_dict[prefix + n + '_a'] = loss.item()
        for n, loss in g_losses_b.items():
            loss_dict[prefix + n + '_b'] = loss.item()
        loss_dict[prefix + 'g_loss'] = g_loss.item()

        if self.global_step > self.training_config.warmup_generator_steps:
            # Train discriminators
            self.toggle_optimizer(opt_d)
            opt_d.zero_grad()
            fake_b = self(real_a)
            fake_a = self(real_b, b2a=True)
            d_losses_a, d_losses_b = self.compute_discriminator_losses(
                real_a, fake_a, real_b, fake_b
            )
            d_loss = sum(d_losses_a.values()) + sum(d_losses_b.values())
            self.manual_backward(d_loss)
            opt_d.step()
            self.untoggle_optimizer(opt_d)

            for n, loss in d_losses_a.items():
                loss_dict[prefix + n + '_a'] = loss.item()
            for n, loss in d_losses_b.items():
                loss_dict[prefix + n + '_b'] = loss.item()
            loss_dict[prefix + 'd_loss'] = d_loss.item()

        self.log_dict(loss_dict, prog_bar=False, on_step=True, on_epoch=True)
    
    def validation_step(self, batch, batch_idx: int):
        real_a, real_b = batch

        fake_b = self(real_a)
        fake_a = self(real_b, b2a=True)

        g_losses_a, g_losses_b = self.compute_generator_losses(
            real_a, fake_a, real_b, fake_b
        )
        g_loss = sum(g_losses_a.values()) + sum(g_losses_b.values())

        d_losses_a, d_losses_b = self.compute_discriminator_losses(
            real_a, fake_a, real_b, fake_b
        )
        d_loss = sum(d_losses_a.values()) + sum(d_losses_b.values())

        loss_dict = dict()
        prefix = 'val/'
        for n, loss in g_losses_a.items():
            loss_dict[prefix + n + '_a'] = loss.item()
        for n, loss in g_losses_b.items():
            loss_dict[prefix + n + '_b'] = loss.item()
        for n, loss in d_losses_a.items():
            loss_dict[prefix + n + '_a'] = loss.item()
        for n, loss in d_losses_b.items():
            loss_dict[prefix + n + '_b'] = loss.item()
        loss_dict.update({
            prefix + 'g_loss': g_loss.item(),
            prefix + 'd_loss': d_loss.item(),
        })

        self.log_dict(loss_dict, on_epoch=True)
        if batch_idx == 0:
            self.log_images(dict(
                real_a=real_a,
                real_b=real_b,
                fake_a=fake_a,
                fake_b=fake_b
            ))

    def configure_optimizers(self):
        assert self.training_config is not None, 'Training config is required for training state!'

        opt_g = torch.optim.AdamW(
            itertools.chain(
                self.generator_ab.parameters(),
                self.generator_ba.parameters()
            ),
            lr=self.training_config.learning_rate,
            weight_decay=self.training_config.weight_decay
        )

        opt_d = torch.optim.AdamW(
            itertools.chain(
                self.discriminator_a.parameters(),
                self.discriminator_b.parameters()
            ),
            lr=self.training_config.learning_rate,
            weight_decay=self.training_config.weight_decay
        )

        return [opt_g, opt_d]

    @torch.inference_mode()
    def log_images(self, im_dict: Dict[str, torch.Tensor]):
        # im_grid_dict = make_image_grid(im_dict)

        for title, im_grid in im_dict.items():
            self.logger.log_image(title, list(im_grid))

    def on_train_batch_end(self, outputs, batch: Any, batch_idx: int) -> None:
        self.store_data(batch, mode='train')

    def on_validation_batch_end(self, outputs, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        self.store_data(batch, mode='val')

    def on_test_batch_end(self, outputs, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        self.store_data(batch, mode='test')

    def store_data(self, 
                   batch: Any, 
                   mode: str):
        if self.photo_images[mode] == None:
            self.photo_images[mode], self.anime_images[mode] = batch
        elif self.photo_images[mode].shape[0] < self.n_image_fid:
            self.photo_images[mode] = torch.cat((self.photo_images[mode], batch[0]), dim=0)
            self.anime_images[mode] = torch.cat((self.anime_images[mode], batch[1]), dim=0)

    def on_train_epoch_end(self) -> None:
        self.compute_fid(mode='train')
    
    def on_validation_epoch_end(self) -> None:
        self.compute_fid(mode='val')

    def on_test_epoch_end(self) -> None:
        self.compute_fid(mode='test')

    def compute_fid(self, mode: str):

        self.fid_photo.reset()
        self.fid_anime.reset()
        
        real_photo = self.photo_images[mode][:self.n_image_fid]
        real_anime = self.anime_images[mode][:self.n_image_fid]

        fake_photo = self(real_anime, b2a=True) * 0.5 + 0.5
        fake_anime = self(real_photo) * 0.5 + 0.5

        self.fid_photo.update(fake_photo, real=False)
        self.fid_photo.update(real_photo, real=True)

        self.fid_anime.update(fake_anime, real=False)
        self.fid_anime.update(real_anime, real=True)

        self.log(mode + '/fid_photo', self.fid_photo.compute(), prog_bar=False, on_step=False, on_epoch=True)
        self.log(mode + '/fid_anime', self.fid_anime.compute(), prog_bar=False, on_step=False, on_epoch=True)

        self.photo_images[mode] = None
        self.anime_images[mode] = None
    