import os
import numpy as np
import torch
from torch.utils.data import TensorDataset
from torchvision import datasets, transforms

from data.cinic10 import fetch_cinic10
from data.tabular_datasets import get_tabular_data

SHAPES = {
    "cifar100": (32, 32, 3),
    "cinic10": (32, 32, 3),
    "texas": (6169,),
    "location": (168,),
}

def get_data(name, augment=True, **kwargs):
    load_path = os.getcwd() + '/data'

    if name == "cifar100":
        if augment:
            normalize = transforms.Normalize(
                mean=[0.5071, 0.4867, 0.4408],
                std=[0.2675, 0.2565, 0.2761],
            )
            train_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize,
            ])
            test_transform = transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])
        else:
            train_transform = transforms.Compose([transforms.ToTensor()])
            test_transform = transforms.Compose([transforms.ToTensor()])

        train_set = datasets.CIFAR100(
            root=load_path,
            train=True,
            transform=train_transform,
            download=True,
        )
        test_set = datasets.CIFAR100(
            root=load_path,
            train=False,
            transform=test_transform,
            download=True,
        )

    elif name == "cinic10":
        # Ensure CINIC-10 exists locally.
        fetch_cinic10(load_path)

        if augment:
            normalize = transforms.Normalize(
                mean=[0.5071, 0.4867, 0.4408],
                std=[0.2675, 0.2565, 0.2761],
            )
            transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            transform = transforms.Compose([transforms.ToTensor()])

        train_set = datasets.ImageFolder(
            root=f'{load_path}/cinic-10-batches-py/train',
            transform=transform,
        )
        test_set = datasets.ImageFolder(
            root=f'{load_path}/cinic-10-batches-py/test',
            transform=transform,
        )
    elif name in ("texas", "location"):
        train_set, test_set = get_tabular_data(name)
    else:
        raise ValueError(
            f"unknown dataset {name}. Supported: cifar100, cinic10, texas, location."
        )

    return train_set, test_set


def get_data_random_sample(num_samples,t):
    print('create the random dataset')

    # Adaptive seed generation based on trial number
    base_seed = 40
    random = base_seed + t
    np.random.seed(random)
    torch.manual_seed(random)
    img_h = 32
    img_w = 32
    output_dim = 2
    # Generate random binary images (num_samples, 1, 32, 32) for CNN.
    X_images = torch.FloatTensor(np.random.randint(2, size=(num_samples, 1, img_h, img_w)))

    # Generate random labels.
    y = torch.LongTensor(np.random.randint(output_dim, size=num_samples))
    
    # Debug: Print shapes
    print(f"Generated data shape: {X_images.shape}")
    print(f"Generated labels shape: {y.shape}")
    
    # Create dataset.
    dataset = TensorDataset(X_images, y)
    train_set = dataset
    test_set = dataset

    return train_set, test_set
