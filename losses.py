import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss
    Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
    """

    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-9

    def forward(self, output1, output2, target, size_average=True):
        distances = (output1 - output2).pow(2).sum(1).sqrt()  # squared distances
        losses = target.float() * distances + (1 + -1 * target).float() * F.relu(self.margin - distances)
        
        x = (target == 0).nonzero(as_tuple=True)[0]
        y = (target == 1).nonzero(as_tuple=True)[0]
        if  x.size() > y.size():
            dist = x.size()
        else:
            dist = y.size()
        a = torch.zeros(64)
        pred =   ((1 + -1 * target).float() * (distances)-(target.float() * distances)).cpu().data
        acc = (pred>0).sum()/a.size()[0]

        return losses.mean() if size_average else losses.sum(), len(output1),acc
    
    
class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1).sqrt()  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1).sqrt()  # .pow(.5)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        
           
        pred = (distance_negative - distance_positive).cpu().data
        acc = (pred>0).sum()/distance_negative.size()[0]
        
        
        return losses.mean() if size_average else losses.sum(), len(anchor), acc
    
class OnlineContrastiveLoss(nn.Module):
    """
    Online Contrastive loss
    Takes a batch of embeddings and corresponding labels.
    Pairs are generated using pair_selector object that take embeddings and targets and return indices of positive
    and negative pairs
    """

    def __init__(self, margin, pair_selector):
        super(OnlineContrastiveLoss, self).__init__()
        self.margin = margin
        self.pair_selector = pair_selector

    def forward(self, embeddings, target):
        positive_pairs, negative_pairs = self.pair_selector.get_pairs(embeddings, target)
        if embeddings.is_cuda:
            positive_pairs = positive_pairs.cuda()
            negative_pairs = negative_pairs.cuda()
        positive_loss = (embeddings[positive_pairs[:, 0]] - embeddings[positive_pairs[:, 1]]).pow(2).sum(1).sqrt()
        negative_dist = (embeddings[negative_pairs[:, 0]] - embeddings[negative_pairs[:, 1]]).pow(2).sum(1).sqrt() #+ self.margin
        negative_loss = F.relu(
            self.margin - (embeddings[negative_pairs[:, 0]] - embeddings[negative_pairs[:, 1]]).pow(2).sum(
                1).sqrt())
        loss = torch.cat([positive_loss, negative_loss], dim=0)
        
        
        pred = (negative_dist - positive_loss).cpu().data
        
       
        acc = (pred>0).sum()/positive_pairs.size()[0]


        return loss.mean(), len(positive_pairs),acc
    
class OnlineTripletLoss(nn.Module):
    """
    Online Triplets loss
    Takes a batch of embeddings and corresponding labels.
    Triplets are generated using triplet_selector object that take embeddings and targets and return indices of
    triplets
    """

    def __init__(self, margin, triplet_selector):
        super(OnlineTripletLoss, self).__init__()
        self.margin = margin
        self.triplet_selector = triplet_selector

    def forward(self, embeddings, target):

        triplets = self.triplet_selector.get_triplets(embeddings, target)

        if embeddings.is_cuda:
            triplets = triplets.cuda()

        ap_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 1]]).pow(2).sum(1).sqrt()  # .pow(.5)
        an_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 2]]).pow(2).sum(1).sqrt()  # .pow(.5)
        losses = F.relu(ap_distances - an_distances + self.margin)
        
        pred =(an_distances- ap_distances).cpu().data
        acc = (pred>0).sum()/ap_distances.size()[0]
  
        return losses.mean(), len(triplets), acc