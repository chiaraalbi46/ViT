""" Run this the first time you want to download CIFAR10/CIFAR100 on Windows.
    Look at this https://github.com/pytorch/vision/issues/5039
"""

import torchvision
import torchvision.transforms as transforms

# these two lines must be executed the first time we download the datasets (then they are no longer needed)
import ssl

ssl._create_default_https_context = ssl._create_unverified_context
#

transform = transforms.Compose(
    [transforms.ToTensor()])

print("CIFAR 10")
root_dir_cifar10 = './cifar10_data'  # where to download the dataset
dataset_c10 = torchvision.datasets.CIFAR10(root=root_dir_cifar10, train=True, download=True, transform=transform)
test_set_c10 = torchvision.datasets.CIFAR10(root=root_dir_cifar10, train=False, download=True, transform=transform)

print("CIFAR 100")
root_dir_cifar100 = './cifar100_data'  # where to download the dataset
dataset_c100 = torchvision.datasets.CIFAR100(root=root_dir_cifar100, train=True, download=True, transform=transform)
test_set_c100 = torchvision.datasets.CIFAR100(root=root_dir_cifar100, train=False, download=True, transform=transform)
