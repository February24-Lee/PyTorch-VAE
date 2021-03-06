import numpy as np
import cv2

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import transforms, datasets

#from pl_bolts.models.self_supervised.simclr.simclr_transforms import SimCLREvalDataTransform, SimCLRTrainDataTransform

class Vanilla_HillshapeDataModule(LightningDataModule):
    def __init__(self, 
                dataset: str,
                data_path:str,
                test_ratio: float,
                img_size: int,
                batch_size: int,
                **kwargs
                 ):
        super().__init__()
        self.dataset = dataset
        self.data_path = data_path
        self.test_ratio = test_ratio
        self.img_size = img_size
        self.batch_size = batch_size

        self.train_num = self.cal_train_data()

    def cal_train_data(self):
        dataset = datasets.ImageFolder(root=self.data_path, 
                                    transform=MySimCLRTrainDataTransform(self.img_size))
        num_train = len(dataset)
        split = int(np.floor(self.test_ratio * num_train))
        return num_train-split

    def train_dataloader(self):
        dataset = datasets.ImageFolder(root=self.data_path, 
                                    transform=MySimCLRTrainDataTransform(self.img_size))
        num_train = len(dataset)
        indices = list(range(num_train))
        split = int(np.floor(self.test_ratio * num_train))

        train_idx = indices[split:]
        train_sampler = SubsetRandomSampler(train_idx)
        self.num_train_imgs = len(train_idx)

        return DataLoader(dataset,
                    num_workers=64,
                    batch_size= self.batch_size,
                    sampler=train_sampler,
                    drop_last=True)

    def val_dataloader(self):
        dataset = datasets.ImageFolder(root=self.data_path,
                                        transform=MySimCLREvalDataTransform(self.img_size))
        num_train = len(dataset)
        indices = list(range(num_train))
        split = int(np.floor(self.test_ratio * num_train))

        val_idx = indices[:split]
        val_sampler = SubsetRandomSampler(val_idx)
        self.num_val_imgs = len(val_idx)
        self.sample_dataloader = DataLoader(dataset,
                                num_workers=64,
                                batch_size= 144,
                                sampler=val_sampler,
                                drop_last=True)
        return self.sample_dataloader


class MySimCLRTrainDataTransform(object):
    def __init__(self, input_height, s=1):
        self.s = s
        self.input_height = input_height
        color_jitter = transforms.ColorJitter(0.8 * self.s, 0.8 * self.s, 0.8 * self.s, 0.2 * self.s)
        data_transforms = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                                transforms.RandomResizedCrop(size=self.input_height),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.RandomApply([color_jitter], p=0.5),
                                                #transforms.RandomGrayscale(p=0.2),
                                                GaussianBlur(kernel_size=int(0.1 * self.input_height)+1),
                                                transforms.ToTensor()])
        self.train_transform = data_transforms

    def __call__(self, sample):
        transform = self.train_transform
        xi = transform(sample)
        xj = transform(sample)
        return xi, xj


class MySimCLREvalDataTransform(object):
    def __init__(self, input_height, s=1):
        self.s = s
        self.input_height = input_height
        self.test_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize(input_height + 10, interpolation=3),
            transforms.CenterCrop(input_height),
            transforms.ToTensor(),
        ])

    def __call__(self, sample):
        transform = self.test_transform
        xi = transform(sample)
        xj = transform(sample)
        return xi, xj


class GaussianBlur(object):
    # Implements Gaussian blur as described in the SimCLR paper
    def __init__(self, kernel_size, min=0.1, max=2.0):
        self.min = min
        self.max = max
        # kernel size is set to be 10% of the image height/width
        self.kernel_size = kernel_size

    def __call__(self, sample):
        sample = np.array(sample)

        # blur the image with a 50% chance
        prob = np.random.random_sample()

        if prob < 0.5:
            sigma = (self.max - self.min) * np.random.random_sample() + self.min
            sample = cv2.GaussianBlur(sample, (self.kernel_size, self.kernel_size), sigma)

        return sample