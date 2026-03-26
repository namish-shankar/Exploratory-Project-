# data/cifar10.py
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np

def get_cifar_loaders(batch_size=128, num_workers=2, split_test=False):
    """
    Returns data loaders for CIFAR-10.
    If split_test=True, returns (trainloader, valloader, testloader).
    If split_test=False, returns (trainloader, testloader).
    """
    # Standard data augmentation for training
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2470, 0.2435, 0.2616)
        ),
    ])

    # Clean, deterministic transforms for validation/testing
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2470, 0.2435, 0.2616)
        ),
    ])

    # pin_memory drastically speeds up data transfer to the GPU if one is available
    pin_mem = torch.cuda.is_available()

    if split_test:
        # 1. Load the underlying data twice with different transforms
        train_data_full = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        val_data_full = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_test)
        
        # 2. Generate a fixed train/val split (45k / 5k is standard for CIFAR-10)
        indices = list(range(len(train_data_full)))
        np.random.seed(42)  # Lock seed so the split is identical across worker processes
        np.random.shuffle(indices)
        train_idx, val_idx = indices[:45000], indices[45000:]
        
        # 3. Create subsets mapping to the correct transforms
        train_subset = Subset(train_data_full, train_idx)
        val_subset = Subset(val_data_full, val_idx)
        
        trainloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_mem)
        valloader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_mem)
        
        # 4. Load the actual test set
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_mem)
        
        return trainloader, valloader, testloader

    else:
        # Standard 2-way split (backward compatibility)
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_mem)
        testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_mem)

        return trainloader, testloader