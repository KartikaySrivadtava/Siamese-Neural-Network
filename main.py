import argparse
from argparse import RawTextHelpFormatter
import os

import torch

from selection import SiameseTrain, TripletNetTrain, OnlinePairSelection, OnlineTripletSelection
from selection import SiameseTest, TripletNetTest, OnlinePairSelectionTest, OnlineTripletSelectionTest
from dataloader import load_train_dataset, load_test_dataset
from draw import draw_training, draw_testing

import numpy as np

names = ['Random Offline Pair', 'Hard Negative Online Pair', 'All Positive Online Pair',
         'Random Offline Triplet', 'All Positive Online Triplet', 'Random Negative Online Triplet',
         'Semi-hard Negative Online Triplet']


def train():
    selection, data, save_dir = \
        args.selection, args.dataset, args.out

    if data == 0:
        dataset = 'MNIST'
    elif data == 1:
        dataset = 'KMNIST'
    else:
        dataset = 'FashionMNIST'

    print('Network:         ' + names[selection])
    print('Dataset:         ' + dataset)
    print('save directiory: ' + save_dir)

    train_dataset = load_train_dataset(dataset)
    test_dataset = load_test_dataset(dataset)

    # 0 = Random Offline Pair
    # 1 = Hard Negative Online Par
    # 2 = All Positive Online Pair
    # 3 = Random Offline Triplet
    # 4 = All Positive Online Triplet
    # 5 = Random Negative Online Triplet
    # 6 = Semi-hard Negative Online Triplet

    if selection == 0:
        model, training_loss, validation_loss, training_acc, validation_acc = SiameseTrain(train_dataset, test_dataset)
    elif selection == 3:
        model, training_loss, validation_loss, training_acc, validation_acc = TripletNetTrain(train_dataset,
                                                                                              test_dataset)
    elif selection == 1 or selection == 2:
        model, training_loss, validation_loss, training_acc, validation_acc = OnlinePairSelection(train_dataset,
                                                                                                  test_dataset,
                                                                                                  selection)
    else:
        model, training_loss, validation_loss, training_acc, validation_acc = OnlineTripletSelection(train_dataset,
                                                                                                     test_dataset,
                                                                                                     selection)

    draw_training(test_dataset, model, training_loss, validation_loss, training_acc, validation_acc, names[selection],
                  dataset, save_dir)

    torch.save(model, save_dir + 'model.pt')


def test():
    weights, selection, data, save_dir = \
        args.model, args.selection, args.dataset, args.out

    if data == 0:
        dataset = 'MNIST'
    elif data == 1:
        dataset = 'KMNIST'
    else:
        dataset = 'FashionMNIST'

    print('Network:         ' + names[selection])
    print('Dataset:         ' + dataset)
    print('save directiory: ' + save_dir)

    if save_dir[-1] != '/':
        save_dir = save_dir + '/'

    test_dataset = load_test_dataset(dataset)
    model = torch.load(weights)

    if selection == 0:
        val_loss, accuracy = SiameseTest(test_dataset, model)
    elif selection == 3:
        val_loss, accuracy = TripletNetTest(test_dataset, model)
    elif selection == 1 or selection == 2:
        val_loss, accuracy = OnlinePairSelectionTest(test_dataset, selection, model)
    else:
        val_loss, accuracy = OnlineTripletSelectionTest(test_dataset, selection, model)

    draw_testing(test_dataset, model, names[selection], dataset, save_dir)
    print('****************************')
    print('validation loss: ' + str(np.mean(val_loss)))
    print('validation accuracy: ' + str(np.round(np.mean(accuracy) * 100, 2)) + '%')
    print('****************************')


def test_arguments():
    if args.selection not in [0, 1, 2, 3, 4, 5, 6]:
        print('please select an existing selection method from 0-6')
        return -1
    if args.dataset not in [0, 1, 2]:
        print('please select an existing dataset from 0-2')
        return -1
    if os.path.exists(args.out) == False:
        print('pleas select an existing output path')
        return -1
    else:
        args.out = os.path.normpath(args.out) + '/'
    if args.model != '':
        if os.path.exists(args.model) == False:
            print('please select an existing model path')
            return -1
    return 0


if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument('--model', default='', type=str, help='path to model for testing')
    parser.add_argument('--selection', default=0, type=int,
                        help='0 = Random Offline Pair \n1 = Hard Negative Online Par \n2 = All Positive Online Pair \n3 = Random Online Triplet \n4 = All Positive Online Triplet \n5 = Random Negative Online Triplet \n6 = Semi-hard Negative Online Triplet')
    parser.add_argument('--dataset', default=0, type=int, help='0 = MNIST\n1 = KMNIST \n2 = FashionMNIST')
    parser.add_argument('--out', default='./output', type=str, help='output path')

    args = parser.parse_args()

    stop = test_arguments()

    if stop == 0:
        if args.model == '':
            train()
        else:
            test()