from torchvision.datasets import FashionMNIST, MNIST, KMNIST
from torchvision import transforms

import torch

mean, std = 0.28604059698879553, 0.35302424451492237


def load_train_dataset(dataset):
    if dataset == 'FashionMNIST':
        FashionMNIST('./data/FashionMNIST', train=True, download=True,
                     transform=transforms.Compose([
                         transforms.ToTensor(),
                         transforms.Normalize((mean,), (std,))
                     ]))

        print('removing one class from the training set')
        generate_train_set('./data/FashionMNIST/FashionMNIST/', 1)  # remove Trouser

        train_dataset = FashionMNIST('./data/FashionMNIST', train=True, download=True,
                                     transform=transforms.Compose([
                                         transforms.ToTensor(),
                                         transforms.Normalize((mean,), (std,))
                                     ]))

    elif dataset == 'MNIST':
        MNIST('./data/MNIST', train=True, download=True,
              transform=transforms.Compose([
                  transforms.ToTensor(),
                  transforms.Normalize((mean,), (std,))
              ]))

        print('removing one class from the training set')
        generate_train_set('./data/MNIST/MNIST/', 0)  # remove zero

        train_dataset = MNIST('./data/MNIST', train=True, download=True,
                              transform=transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.Normalize((mean,), (std,))
                              ]))

    elif dataset == 'KMNIST':
        KMNIST('./data/KMNIST', train=True, download=True,
               transform=transforms.Compose([
                   transforms.ToTensor(),
                   transforms.Normalize((mean,), (std,))
               ]))

        print('removing one class from the training set')
        generate_train_set('./data/KMNIST/KMNIST/', 0)  # remove o

        train_dataset = KMNIST('./data/KMNIST', train=True, download=True,
                               transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize((mean,), (std,))
                               ]))

    return train_dataset


def load_test_dataset(dataset):
    if dataset == 'FashionMNIST':
        test_dataset = FashionMNIST('./data/FashionMNIST', train=False, download=True,
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((mean,), (std,))
                                    ]))
    elif dataset == 'MNIST':
        test_dataset = MNIST('./data/MNIST', train=False, download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize((mean,), (std,))
                             ]))
    elif dataset == 'KMNIST':
        test_dataset = KMNIST('./data/KMNIST', train=False, download=True,
                              transform=transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.Normalize((mean,), (std,))
                              ]))

    return test_dataset


def load_train_dataset10(dataset):
    if dataset == 'FashionMNIST':
        train_dataset = FashionMNIST('./data/FashionMNIST10', train=True, download=True,
                                     transform=transforms.Compose([
                                         transforms.ToTensor(),
                                         transforms.Normalize((mean,), (std,))
                                     ]))
    elif dataset == 'MNIST':
        train_dataset = MNIST('./data/MNIST10', train=True, download=True,
                              transform=transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.Normalize((mean,), (std,))
                              ]))

    elif dataset == 'KMNIST':

        train_dataset = KMNIST('./data/KMNIST10', train=True, download=True,
                               transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize((mean,), (std,))
                               ]))

    return train_dataset


def generate_train_set(path, c):
    data, label = torch.load(path + '/processed/training.pt')

    x = (label == c).nonzero(as_tuple=True)[0]
    count = 0
    for i in x:
        label = torch.cat([label[0:i - count], label[i - count + 1:]])
        data = torch.cat([data[0:i - count], data[i - count + 1:]])
        count = count + 1
    torch.save((data, label), path + '/processed/training.pt')

    print(len(data))