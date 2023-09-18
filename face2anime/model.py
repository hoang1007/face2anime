from typing import Dict, Optional
import itertools

from lightning.pytorch import LightningModule
import torch
from torch import nn
import torch.nn.functional as F

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
        warmup_generator_steps: int = 200
    ):
        self.lambda_ab = lambda_ab
        self.lambda_ba = lambda_ba
        self.identity_loss_weight = identity_loss_weight
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.use_lsgan = use_lsgan
        self.warmup_generator_steps = warmup_generator_steps


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

        self.generator_ab = generator_ab
        self.discriminator_b = discriminator_b

        self.generator_ba = generator_ba
        self.discriminator_a = discriminator_a

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
