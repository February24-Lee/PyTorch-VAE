from warnings import warn
import numpy as np
import cv2

import pytorch_lightning as pl
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torchvision import transforms

try:
    from torchvision.models import densenet
except ImportError:
    warn('torch vision not installed yet')

from pl_bolts.callbacks.self_supervised import SSLOnlineEvaluator
from pl_bolts.losses.self_supervised_learning import nt_xent_loss
from pl_bolts.models.self_supervised.evaluator import Flatten
from pl_bolts.models.self_supervised.resnets import resnet50_bn
#from pl_bolts.models.self_supervised.simclr.simclr_transforms import SimCLREvalDataTransform, SimCLRTrainDataTransform
from pl_bolts.optimizers.lars_scheduling import LARSWrapper
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR

_LATENT_DIM = 32
_HIDDEN_DIM = 10
_OUTPUT_DIM = 5
_INCHANNEL = 1

class Projection(nn.Module):
    def __init__(self, 
                input_dim=_LATENT_DIM,
                hidden_dim=_HIDDEN_DIM, 
                output_dim=_OUTPUT_DIM):
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.model = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            Flatten(),
            nn.Linear(self.input_dim, self.hidden_dim, bias=True),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim, bias=False))

    def forward(self, x):
        x = self.model(x)
        return F.normalize(x, dim=1)

class SimCLR_sate(pl.LightningModule):
    def __init__(self,
                batch_size,
                num_samples,
                warmup_epochs=10,
                lr=1e-4,
                opt_weight_decay=1e-6,
                loss_temperature=0.5,
                **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.nt_xent_loss = nt_xent_loss
        self.encoder = self.init_encoder()
        self.Projection = Projection()
        
    def init_encoder(self):
        modules = []
        hidden_dims = [32, 32, 32, 32]
        in_channels = _INCHANNEL

        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                            kernel_size= 4, stride=2, padding= 1),
                    nn.LeakyReLU())
            )
            in_channels = h_dim
        '''
        modules.append(
            nn.Sequential(
                nn.Flatten(start_dim=1),
                nn.Linear(hidden_dims[-1]*16, 256),
                nn.Linear(256, _LATENT_DIM))
                )
        '''
        return nn.Sequential(*modules)

    def exclude_from_wt_decay(self, named_params, weight_decay, skip_list=['bias', 'bn']):
        params = []
        excluded_params = []

        for name, param in named_params:
            if not param.requires_grad:
                continue
            elif any(layer_name in name for layer_name in skip_list):
                excluded_params.append(param)
            else:
                params.append(param)

        return [
            {'params': params, 'weight_decay': weight_decay},
            {'params': excluded_params, 'weight_decay': 0.}
        ]

    def setup(self, stage):
        global_batch_size = self.trainer.world_size * self.hparams.batch_size
        self.train_iters_per_epoch = self.hparams.num_samples // global_batch_size

    def configure_optimizers(self):
        parameters = self.exclude_from_wt_decay(
            self.named_parameters(),
            weight_decay=self.hparams.opt_weight_decay
        )

        optimizer = LARSWrapper(Adam(parameters, lr=self.hparams.lr))

        self.hparams.warmup_epochs = self.hparams.warmup_epochs * self.train_iters_per_epoch
        max_epochs = self.trainer.max_epochs * self.train_iters_per_epoch

        linear_warmup_cosine_decay = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=self.hparams.warmup_epochs,
            max_epochs=max_epochs,
            warmup_start_lr=0,
            eta_min=0
        )

        scheduler = {
            'scheduler': linear_warmup_cosine_decay,
            'interval': 'step',
            'frequency': 1 
        }

        return [optimizer], [scheduler]

    def forward(self, x):
        if isinstance(x, list):
            x = x[0]

        result = self.encoder(x)
        if isinstance(result, list):
            result = result[-1]
        return result

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx)

        result = pl.TrainResult(minimize=loss)
        result.log('train_loss', loss, on_epoch=True)
        return result

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx)

        result = pl.EvalResult(checkpoint_on=loss)
        result.log('avg_val_loss', loss)
        return result

    def shared_step(self, batch, batch_idx):
        (img1, img2), y = batch

        h1 = self.encoder(img1)
        h2 = self.encoder(img2)

        if isinstance(h1, list):
            h1 = h1[-1]
            h2 = h2[-1]

        z1 = self.Projection(h1)
        z2 = self.Projection(h2)

        loss = self.nt_xent_loss(z1, z2, self.hparams.loss_temperature)
        return loss
    