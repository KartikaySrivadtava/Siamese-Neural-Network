import torch
from torch.optim import lr_scheduler
import torch.optim as optim

from datasets import SiameseMNIST, TripletMNIST, BalancedBatchSampler
from networks import EmbeddingNet, SiameseNet, TripletNet
from trainer import fit, test_epoch

from losses import ContrastiveLoss, TripletLoss, OnlineContrastiveLoss, OnlineTripletLoss
from utils import AllPositivePairSelector, HardNegativePairSelector  # Strategies for selecting pairs within a minibatch
from utils import AllTripletSelector, RandomNegativeTripletSelector, SemihardNegativeTripletSelector  # Strategies for selecting triplets within a minibatch

batch_size = 128
cuda = torch.cuda.is_available()
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}


def SiameseTrain(train_dataset, test_dataset):
    # Set up data loaders
    train_dataset = SiameseMNIST(train_dataset)  # Returns pairs of images and target same/different
    test_dataset = SiameseMNIST(test_dataset)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                                       **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                                      **kwargs)
    # Set up the network and training parameters
    embedding_net = EmbeddingNet()
    model = SiameseNet(embedding_net)
    if cuda:
        model.cuda()
    margin = 1.
    loss_fn = ContrastiveLoss(margin)
    lr = 1e-3
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
    n_epochs = 20
    log_interval = 500
    
    training_loss, validation_loss, training_acc, validation_acc = fit(train_loader, test_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval)

    return model, training_loss, validation_loss, training_acc, validation_acc

def TripletNetTrain(train_dataset, test_dataset):
    triplet_train_dataset = TripletMNIST(train_dataset)  # Returns triplets of images
    triplet_test_dataset = TripletMNIST(test_dataset)

    triplet_train_loader = torch.utils.data.DataLoader(triplet_train_dataset, batch_size=batch_size, shuffle=True,
                                                       **kwargs)
    triplet_test_loader = torch.utils.data.DataLoader(triplet_test_dataset, batch_size=batch_size, shuffle=False,
                                                      **kwargs)

    # Set up the network and training parameters

    margin = 1.
    embedding_net = EmbeddingNet()
    model = TripletNet(embedding_net)
    if cuda:
        model.cuda()
    loss_fn = TripletLoss(margin)
    lr = 1e-3
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
    n_epochs = 20
    log_interval = 500
    training_loss, validation_loss, training_acc, validation_acc = fit(triplet_train_loader, triplet_test_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval)
    
    return model, training_loss, validation_loss, training_acc, validation_acc

def OnlinePairSelection(train_dataset, test_dataset, pairSelection):
    train_batch_sampler = BalancedBatchSampler(train_dataset.train_labels, n_classes=9, n_samples=25)
    test_batch_sampler = BalancedBatchSampler(test_dataset.test_labels, n_classes=10, n_samples=25)

    online_train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=train_batch_sampler, **kwargs)
    online_test_loader = torch.utils.data.DataLoader(test_dataset, batch_sampler=test_batch_sampler, **kwargs)

    # Set up the network and training parameters
    

    margin = 1.
    embedding_net = EmbeddingNet()
    model = embedding_net
    if cuda:
        model.cuda()
        
    
    # 1 = Hard Negative Online Par 
    # 2 = All Positive Online Pair 
    if pairSelection == 2:
        loss_fn = OnlineContrastiveLoss(margin, AllPositivePairSelector())
    else:
        loss_fn = OnlineContrastiveLoss(margin, HardNegativePairSelector())
    lr = 1e-3
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
    n_epochs = 20
    log_interval = 250
    training_loss, validation_loss, training_acc, validation_acc = fit(online_train_loader, online_test_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval)
    
    return model, training_loss, validation_loss, training_acc, validation_acc

def OnlineTripletSelection(train_dataset, test_dataset, tripletSelection):
    # We'll create mini batches by sampling labels that will be present in the mini batch and number of examples from each class
    train_batch_sampler = BalancedBatchSampler(train_dataset.train_labels, n_classes=9, n_samples=25)
    test_batch_sampler = BalancedBatchSampler(test_dataset.test_labels, n_classes=10, n_samples=25)

    online_train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=train_batch_sampler, **kwargs)
    online_test_loader = torch.utils.data.DataLoader(test_dataset, batch_sampler=test_batch_sampler, **kwargs)

    # Set up the network and training parameters
    margin = 1.
    embedding_net = EmbeddingNet()
    model = embedding_net
    if cuda:
        model.cuda()
        
    # 4 = All Positive Online Triplet 
    # 5 = Random Negative Online Triplet 
    # 6 = Semi-hard Negative Online Triplet
    
    if tripletSelection == 4:
        loss_fn = OnlineTripletLoss(margin, AllTripletSelector())
    elif tripletSelection == 5:
        loss_fn = OnlineTripletLoss(margin, RandomNegativeTripletSelector(margin))
    else:
        loss_fn = OnlineTripletLoss(margin, SemihardNegativeTripletSelector(margin))
    lr = 1e-3
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
    n_epochs = 20
    log_interval = 150
    training_loss, validation_loss, training_acc, validation_acc = fit(online_train_loader, online_test_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval)
    
    return model, training_loss, validation_loss, training_acc, validation_acc


def SiameseTest(test_dataset, model):
    test_dataset = SiameseMNIST(test_dataset)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                                      **kwargs)
    if cuda:
        model.cuda()
    margin = 1.
    loss_fn = ContrastiveLoss(margin)

    val_loss,  _ , accuracy = test_epoch(test_loader, model, loss_fn, cuda, [])

    val_loss/=len(test_loader)
    return val_loss,  accuracy

def TripletNetTest(test_dataset, model):
    triplet_test_dataset = TripletMNIST(test_dataset)
    test_loader = torch.utils.data.DataLoader(triplet_test_dataset, batch_size=batch_size, shuffle=False,
                                                      **kwargs)

    # Set up the network and training parameters
    margin = 1.
    if cuda:
        model.cuda()
    loss_fn = TripletLoss(margin)

    val_loss,  _ , accuracy = test_epoch(test_loader, model, loss_fn, cuda, [])

    val_loss/=len(test_loader)
    return val_loss,  accuracy

def OnlinePairSelectionTest(test_dataset, pairSelection, model):
    test_batch_sampler = BalancedBatchSampler(test_dataset.test_labels, n_classes=9, n_samples=25)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_sampler=test_batch_sampler, **kwargs)

    # Set up the network and training parameters
    

    margin = 1.
    if cuda:
        model.cuda()
        
    
    # 1 = Hard Negative Online Par 
    # 2 = All Positive Online Pair 
    if pairSelection == 2:
        loss_fn = OnlineContrastiveLoss(margin, AllPositivePairSelector())
    else:
        loss_fn = OnlineContrastiveLoss(margin, HardNegativePairSelector())

    val_loss,  _ , accuracy = test_epoch(test_loader, model, loss_fn, cuda, [])

    val_loss/=len(test_loader)
    return val_loss,  accuracy


def OnlineTripletSelectionTest(test_dataset, tripletSelection, model):
    
    test_batch_sampler = BalancedBatchSampler(test_dataset.test_labels, n_classes=9, n_samples=25)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_sampler=test_batch_sampler, **kwargs)

    margin = 1.
    if cuda:
        model.cuda()
        
    # 4 = All Positive Online Triplet 
    # 5 = Random Negative Online Triplet 
    # 6 = Semi-hard Negative Online Triplet
    
    if tripletSelection == 4:
        loss_fn = OnlineTripletLoss(margin, AllTripletSelector())
    elif tripletSelection == 5:
        loss_fn = OnlineTripletLoss(margin, RandomNegativeTripletSelector(margin))
    else:
        loss_fn = OnlineTripletLoss(margin, SemihardNegativeTripletSelector(margin))

    val_loss,  _ , accuracy = test_epoch(test_loader, model, loss_fn, cuda, [])
    
    val_loss/=len(test_loader)
    return val_loss,  accuracy

