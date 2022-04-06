""" Select the dataset and the associated dataloaders """

import torchvision
import torchvision.transforms as transforms
import torch
from torch.utils.data import random_split
import math


def get_dataset(ds, hyperparams, val_perc):
    transform = transforms.Compose(
        [transforms.ToTensor()])

    if hyperparams['rand_aug_numops'] is not None and hyperparams['rand_aug_magn'] is not None:
        print("Add Rand Augmentation")
        transform = transforms.Compose([
            transforms.RandAugment(num_ops=int(hyperparams['rand_aug_numops']),
                                   magnitude=int(hyperparams['rand_aug_magn'])),
            transforms.ToTensor()])

    if 'cifar' in ds:
        if 'cifar100' in ds:
            print("CIFAR 100")
            dataset = torchvision.datasets.CIFAR100(root=ds, train=True, download=False, transform=transform)
            test_set = torchvision.datasets.CIFAR100(root=ds, train=False, download=False,
                                                     transform=transforms.ToTensor())
        else:
            print("CIFAR 10")
            dataset = torchvision.datasets.CIFAR10(root=ds, train=True, download=False, transform=transform)
            test_set = torchvision.datasets.CIFAR10(root=ds, train=False, download=False,
                                                    transform=transforms.ToTensor())

        if val_perc > 0:
            print("Split the dataset in train and validation set")
            val_size = round((int(val_perc) / 100) * len(dataset))
            train_size = len(dataset) - val_size

            train_set, validation_set = random_split(dataset, [train_size, val_size])

            train_loader = torch.utils.data.DataLoader(train_set, batch_size=hyperparams['batch_size'], shuffle=True)
            validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=hyperparams['batch_size'],
                                                            shuffle=False)
            test_loader = validation_loader

            # Test correctness
            assert (math.ceil(len(train_set) / hyperparams['batch_size']) == len(train_loader)), \
                "Train loader len must be equal to len(train_dataset) / batch_size"

            assert (math.ceil(len(validation_set) / hyperparams['batch_size']) == len(validation_loader)), \
                "Validation loader len must be equal to len(validation_set) / batch_size"

        else:
            print("No validation set used")
            train_loader = torch.utils.data.DataLoader(dataset, batch_size=hyperparams['batch_size'], shuffle=True)
            test_loader = torch.utils.data.DataLoader(test_set, batch_size=hyperparams['batch_size'], shuffle=False)

            # Test correctness
            assert (math.ceil(len(dataset) / hyperparams['batch_size']) == len(train_loader)), \
                "Train loader len must be equal to len(dataset) / batch_size"

            assert (math.ceil(len(test_set) / hyperparams['batch_size']) == len(test_loader)), \
                "Test loader len must be equal to len(test_dataset) / batch_size"

    else:
        print("IMAGENET")
        dataset = torchvision.datasets.ImageFolder(root=ds + "/train", transform=transform)
        test_set = torchvision.datasets.ImageFolder(root=ds + "/validation", transform=transforms.ToTensor())

        # assume no split here

        train_loader = torch.utils.data.DataLoader(dataset=dataset, shuffle=True, batch_size=hyperparams['batch_size'])
        test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=hyperparams['batch_size'], shuffle=False)

        # Test correctness
        assert (math.ceil(len(dataset) / hyperparams['batch_size']) == len(train_loader)), \
            "Train loader len must be equal to len(dataset) / batch_size"

        assert (math.ceil(len(test_set) / hyperparams['batch_size']) == len(test_loader)), \
            "Test loader len must be equal to len(test_dataset) / batch_size"

    return train_loader, test_loader


if __name__ == '__main__':
    import json

    with open('hyper_prova.json') as json_file:
        hyperparams = json.load(json_file)
    print(hyperparams)

    train_l, test_l = get_dataset('./cifar100_data', hyperparams, 0)

