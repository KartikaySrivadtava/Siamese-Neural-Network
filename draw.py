import inference
import dataloader

import numpy as np
import matplotlib.pyplot as plt

MNIST_CLASSES = ['0 - zero', '1 - one', '2 - two', '3 - three', '4 - four',
               '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine']

KMNIST_CLASSES = ['o', 'ki', 'su', 'tsu', 'na', 'ha', 'ma', 'ya', 're', 'wo']

FMNIST_CLASSES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal',
               'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def draw_training(test_dataset, model, training_loss, validation_loss, training_acc, validation_acc, name, dataset, save_dir):
    
    titel = name + '\n' +  dataset + '\n'

    
    draw_figure([training_loss], [name], save_dir + 'training_loss.png', titel + 'training loss', 'Epoch', 'Loss')
    draw_figure([validation_loss], [name], save_dir + 'validation.png', titel + 'validation loss', 'Epoch', 'Loss')
    draw_figure([training_acc], [name], save_dir + 'training_accuracy.png', titel + 'training accuracy', 'Epoch', 'Loss')
    draw_figure([validation_acc], [name], save_dir + 'validation_accuracy.png', titel + 'validation accuracy', 'Epoch', 'Loss')
    
    train_dataset10 = dataloader.load_train_dataset10(dataset)
    
    embeddings, labels= inference.do_inference(train_dataset10, model)
    c = 0
    if dataset == 'FashionMNIST':
        c = 1
    draw_image(embeddings, labels, dataset, titel + 'Training' , save_dir + 'train_classes.png', white=c)
    
    embeddings = []
    labels = []
    embeddings, labels= inference.do_inference(test_dataset, model)
    draw_image(embeddings, labels, dataset, titel + 'Validation' , save_dir + 'val_classes.png', white=c)
    
    return

def draw_testing(test_dataset, model, name, dataset, save_dir):    
    titel = name + '\n' +  dataset + '\n'
    c = 0
    if dataset == 'FashionMNIST':
        c = 1
        
    embeddings, labels= inference.do_inference(test_dataset, model)
    draw_image(embeddings, labels, dataset, titel + 'Validation' , save_dir + 'val_classes.png', white=c)
    
    return
    


def draw_figure(inputs, labels, outname, title, x, y):
    fig, ax = plt.subplots(1, 1, figsize=(10,5))

    plt.title(title)#, fontsize=40)
    #ax.set_ylabel(y, fontsize=30)
    #ax.set_xlabel(x, fontsize=30)
    #ax.tick_params(labelsize=30)
    
    for i in range(len(inputs)):
        plt.plot(inputs[i], label=labels[i])

    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)#, fontsize=30)
    plt.savefig(outname, dpi=300, bbox_inches='tight')
    
    
    
    
def draw_image(embeddings, labels, dataset, title, outname, white = -1):
    
    colors = ['red', 'blue', 'green', 'orange',
              'yellow', 'purple', 'brown', 'pink',
              'cyan', 'grey']
    
    if white == 1:
        colors = ['blue', 'red', 'green', 'orange',
              'yellow', 'purple', 'brown', 'pink',
              'cyan', 'grey']

    if dataset == 'MNIST':
        classes = MNIST_CLASSES
    elif dataset == 'KMNIST':
        classes = KMNIST_CLASSES
    else:
        classes = FMNIST_CLASSES
        
    fig, ax = plt.subplots(1, 1, figsize=(10,10))
        
    plt.title(title)#, fontsize=40)
    #ax.set_ylabel('y-dist', fontsize=30)
    #ax.set_xlabel('x-dist', fontsize=30)
    #ax.tick_params(labelsize=30)
    
    for i in range(10):
        inds = np.where(labels==i)[0]
        if i != white:
            plt.scatter(embeddings[inds,0], embeddings[inds,1], alpha=0.5, color=colors[i])
    inds = np.where(labels==white)[0]
    plt.scatter(embeddings[inds,0], embeddings[inds,1], alpha=0.5, color=colors[white])
    plt.legend(classes, bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)#, fontsize=30)
    plt.savefig(outname, dpi=300, bbox_inches='tight') 
    