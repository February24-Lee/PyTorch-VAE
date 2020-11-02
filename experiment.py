import math
import torch
from pathlib import Path
from torch import optim
from models import BaseVAE
from models.types_ import *
from utils import data_loader
import pytorch_lightning as pl
from torchvision import transforms, datasets
import torchvision.utils as vutils
from torchvision.datasets import CelebA
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np

class VAEXperiment(pl.LightningModule):

    def __init__(self,
                 vae_model: BaseVAE,
                 params: dict) -> None:
        super(VAEXperiment, self).__init__()

        self.model = vae_model
        self.params = params
        self.curr_device = None
        self.hold_graph = False
        self.z_list = []
        self.save_hyperparameters()

        try:
            self.hold_graph = self.params['retain_first_backpass']
        except:
            pass
        
        
    def setup(self, stage='fit'):
        #self.logger.experiment.log_parameters(self.params)
        self.logger.experiment.set_model_graph(str(self.model))
        return

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input, **kwargs)

    def training_step(self, batch, batch_idx, optimizer_idx = 0):
        real_img, labels = batch
        self.curr_device = real_img.device

        results = self.forward(real_img, labels = labels)
        train_loss = self.model.loss_function(*results,
                                              M_N = self.params['batch_size']/ self.num_train_imgs,
                                              optimizer_idx=optimizer_idx,
                                              batch_idx = batch_idx)
        #for key in train_loss:
        #    self.log_metric(key, train_loss[key], on_step=True)
        self.logger.experiment.log_metrics({key: val.item() for key, val in train_loss.items()}, step=self.global_step)
        return train_loss

    def validation_step(self, batch, batch_idx, optimizer_idx = 0):
        real_img, labels = batch
        self.curr_device = real_img.device

        results = self.forward(real_img, labels = labels)
        val_loss = self.model.loss_function(*results,
                                            M_N = self.params['batch_size']/ self.num_val_imgs,
                                            optimizer_idx = optimizer_idx,
                                            batch_idx = batch_idx)
        # --- validation latent vector check
        self.z_list.append(results[-1].to('cpu').detach().numpy())

        return val_loss

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        #self.log('avg_val_loss', avg_loss, on_epoch=True)
        self.logger.experiment.log_metric('avg_val_loss', avg_loss, epoch=self.current_epoch)
        if self.current_epoch % self.params['frequency_img_save'] == 0:
            self.sample_images()
        #z_list = np.array(self.z_list).reshape(-1,self.model.latent_dim)
        #for index in range(z_list.shape[-1]):
        #    self.logger.experiment.log_histogram_3d(z_list[:,index], name='latent_vector {}'.format(index), step=self.global_step)
        #self.z_list = []
        return 

    def sample_images(self):
        # Get sample reconstruction image
        test_input, test_label = next(iter(self.sample_dataloader))
        test_input = test_input.to(self.curr_device)
        test_label = test_label.to(self.curr_device)
        recons = self.model.generate(test_input, labels = test_label)
        #self.logger.experiment.log_image(vutils.make_grid(test_input.data, normalize=True, nrow=12).permute(1,2,0),
        #                                name='input_img',
        #                                step=self.current_epoch)
        #self.logger.experiment.log_image(vutils.make_grid(recons.data, normalize=True, nrow=12).permute(1,2,0),
        #                                name = 'recons_img',
        #                                step = self.current_epoch)
        Path(f"{self.logger.save_dir}{self.logger.name}/{self.logger.version}/").mkdir(parents=True, exist_ok=True)
        vutils.save_image(torch.cat([recons.data, test_input.data], dim=0),
                          f"{self.logger.save_dir}{self.logger.name}/{self.logger.version}/"
                          f"recons_{self.logger.name}_{self.current_epoch}.png",
                          normalize=True,
                          nrow=12)

        # vutils.save_image(test_input.data,
        #                   f"{self.logger.save_dir}{self.logger.name}/version_{self.logger.version}/"
        #                   f"real_img_{self.logger.name}_{self.current_epoch}.png",
        #                   normalize=True,
        #                   nrow=12)

        try:
            samples = self.model.sample(144,
                                        self.curr_device,
                                        labels = test_label)
            #self.logger.experiment.log_image(vutils.make_grid(samples, normalize=True, nrow=12),
            #                    name = 'sample image',
            #                    step = self.current_epoch)
            vutils.save_image(samples.cpu().data,
                              f"{self.logger.save_dir}{self.logger.name}/{self.logger.version}/"
                              f"{self.logger.name}_{self.current_epoch}.png",
                              normalize=True,
                              nrow=12)
        except:
            pass


        del test_input, recons #, samples

    def configure_optimizers(self):

        optims = []
        scheds = []

        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.params['LR'],
                               weight_decay=self.params['weight_decay'])
        optims.append(optimizer)
        # Check if more than 1 optimizer is required (Used for adversarial training)
        try:
            if self.params['LR_2'] is not None:
                optimizer2 = optim.Adam(getattr(self.model,self.params['submodel']).parameters(),
                                        lr=self.params['LR_2'])
                optims.append(optimizer2)
        except:
            pass

        try:
            if self.params['scheduler_gamma'] is not None:
                scheduler = optim.lr_scheduler.ExponentialLR(optims[0],
                                                             gamma = self.params['scheduler_gamma'])
                scheds.append(scheduler)

                # Check if another scheduler is required for the second optimizer
                try:
                    if self.params['scheduler_gamma_2'] is not None:
                        scheduler2 = optim.lr_scheduler.ExponentialLR(optims[1],
                                                                      gamma = self.params['scheduler_gamma_2'])
                        scheds.append(scheduler2)
                except:
                    pass
                return optims, scheds
        except:
            return optims

    @data_loader
    def train_dataloader(self):
        transform = self.data_transforms()

        if self.params['dataset'] == 'celeba':
            dataset = CelebA(root = self.params['data_path'],
                             split = "train",
                             transform=transform,
                             download=False)
        elif self.params['dataset'] in  ['my_celeba', 'satellite_hill', 'satellite_rgb']:
            dataset = datasets.ImageFolder(root=self.params['data_path'],
                                        transform=transform)
            num_train = len(dataset)
            indices = list(range(num_train))
            split = int(np.floor(self.params['test_ratio'] * num_train))

            train_idx = indices[split:]
            train_sampler = SubsetRandomSampler(train_idx)
            self.num_train_imgs = len(train_idx)

            return DataLoader(dataset,
                          batch_size= self.params['batch_size'],
                          sampler=train_sampler,
                          drop_last=True,
                          num_workers=self.params['num_workers'])
        else:
            raise ValueError('Undefined dataset type')

        self.num_train_imgs = len(dataset)
        return DataLoader(dataset,
                          batch_size= self.params['batch_size'],
                          shuffle = True,
                          drop_last=True,
                          num_workers=self.params['num_workers'])

    @data_loader
    def val_dataloader(self):
        transform = self.data_transforms()

        if self.params['dataset'] == 'celeba':
            self.sample_dataloader =  DataLoader(CelebA(root = self.params['data_path'],
                                                        split = "test",
                                                        transform=transform,
                                                        download=False),
                                                 batch_size= 144,
                                                 shuffle = True,
                                                 drop_last=True,
                                                 num_workers=self.params['num_workers'])
            self.num_val_imgs = len(self.sample_dataloader)

        elif self.params['dataset'] in ['my_celeba', 'satellite_hill', 'satellite_rgb']:
            dataset = datasets.ImageFolder(root=self.params['data_path'],
                                        transform=transform)
            num_train = len(dataset)
            indices = list(range(num_train))
            split = int(np.floor(self.params['test_ratio'] * num_train))

            val_idx = indices[:split]
            val_sampler = SubsetRandomSampler(val_idx)
            self.num_val_imgs = len(val_idx)
            self.sample_dataloader = DataLoader(dataset,
                                    batch_size= 144,
                                    sampler=val_sampler,
                                    drop_last=True,
                                    num_workers=self.params['num_workers'])
            return self.sample_dataloader
        else:
            raise ValueError('Undefined dataset type')

        return self.sample_dataloader

    def data_transforms(self):

        SetRange = transforms.Lambda(lambda X: 2 * X - 1.)
        SetScale = transforms.Lambda(lambda X: X/X.sum(0).expand_as(X))

        if self.params['dataset'] in ['celeba', 'my_celeba', 'satellite_rgb']:
            transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                            transforms.CenterCrop(148),
                                            transforms.Resize(self.params['img_size']),
                                            transforms.ToTensor(),
                                            SetRange])
        elif self.params['dataset'] in ['satellite_hill']:
            transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.CenterCrop(148),
                                            transforms.Resize(self.params['img_size']),
                                            transforms.ToTensor(),
                                            SetRange])
        else:
            raise ValueError('Undefined dataset type')
        return transform

