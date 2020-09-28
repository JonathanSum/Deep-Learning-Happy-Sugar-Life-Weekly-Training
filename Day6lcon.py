import cv2
import numpy as np
import torchvision.transforms as transforms

from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateLogger

from pl_bolts.models.self_supervised.resnets import resnet50_bn
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from pl_bolts.datamodules import CIFAR10DataModule, STL10DataModule, ImagenetDataModule
from pl_bolts.metrics import mean, accuracy

from pl_bolts.models.self_supervised.evaluator import Flatten
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization, stl10_normalization
from pl_bolts.transforms.dataset_normalizations import imagenet_normalization
from pl_bolts.optimizers import LARSWrapper

class SimCLRTrainDataTransform(object):
    def __init__(
        self, input_height: int = 224,
        gaussian_blur: bool = False,
        jitter_strength: float = 1.,
        normalize: optional[transforms.Normalize] = None
    ) -> None:

        self.jitter_strength = jitter_strength
        self.input_height = input_height
        self.gaussian_blur = gaussian_blur
        self.normalize = normalize

        self.color_jitter = transforms.ColorJitter(
            0.8 * self.jitter_strength,
            0.8 * self.jitter_strength,
            0.8 * self.jitter_strength,
            0.2 * self.jitter_strength,
        )

        data_transforms = [
            transforms.RandomResizedCrop(size = self.input_height),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([self.color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2)
        ]
        if self.gaussian_blur:
            data_transforms.append(GaussianBlur(kernel_size = int(0.1 * self.input_height, p=0.5)))
        
        data_transforms.append(transforms.ToTensor())

        if self.normalize:
            data_transforms.append(normalize)

        self.train_transform = transforms.COmpose(data_transforms)

    def __call(self, sample):
        transform = self.train_transform

        xi = transforms(sample)
        xj = transforms(sample)

        return xi, xj



















