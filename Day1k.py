import numpy as np
import torchvision
import torchvision
import matplotlib.pyplot as plt
%matplotlib inline
import logging
logging.basicConfig(
        format = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
        datefmt = "%m/%d/%Y %H:%M:%S",
        level = logging.INFO,
)

from mingpt.utils import set_seed
set_seed(42)
root = './'
train_data = torchvision.datasets.CIFAR10(root, train = True, transform=None, target_transform=None, download = True)
test_data = torchvision.datasets.CIFAR10(root, train = False, transform=None, target_transform=None, download = True)
)
print(len(train_data), len(test_data))