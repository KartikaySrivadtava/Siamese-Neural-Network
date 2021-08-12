import torch
import numpy as np

mean, std = 0.28604059698879553, 0.35302424451492237
batch_size = 256

cuda = torch.cuda.is_available()
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
batch_size = 128

def get_dataloader(dataset):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, **kwargs)
    return dataloader




def do_inference(dataset, model):
    dataloader = get_dataloader(dataset)
    with torch.no_grad():
        model.eval()
        embeddings = np.zeros((len(dataloader.dataset), 2))
        labels = np.zeros(len(dataloader.dataset))
        k = 0
  
        for images, target in dataloader:
            if cuda:
                images = images.cuda()
            embeddings[k:k+len(images)] = model.get_embedding(images).data.cpu().numpy()
            labels[k:k+len(images)] = target.numpy()
            k += len(images)
    return embeddings, labels
            
            
