############################
# written by Trang H. Tran and Lam M. Nguyen
############################

"""
Load Data 

dataname: 'fashionmnist' or 'cifar10' 
"""

import torch
import torchvision 
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def load_data (dataname, batch):
    print('--- Loading data', dataname, 'with batch size', batch)

    if (dataname == 'cifar10'):
        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        dataset = torchvision.datasets.CIFAR10
    
    elif (dataname == 'fashionmnist'):
        normalize = transforms.Normalize((0.2860,), (0.3530,))
        dataset = torchvision.datasets.FashionMNIST

    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize ])
    
    train_set = dataset(root='./data', train=True, transform=transform, download=True) 
    test_set = dataset(root='./data', train=False, transform=transform, download=True)

    train_loader = DataLoader(train_set, batch_size=batch, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch, shuffle=False)

    return train_loader, test_loader