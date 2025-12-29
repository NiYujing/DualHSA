import torch
import torch.nn as nn
import torch.nn.functional as F


def nll_loss(output, target):
    return F.nll_loss(output, target)

def bce_withlogits_loss(output, target):
    return F.binary_cross_entropy_with_logits(output, target)

def bce_loss(output, target):
    return F.binary_cross_entropy(output, target)



def focal_loss_with_logits(output, target, alpha=0.25, gamma=2.0, reduction='mean'):
    bce_loss = F.binary_cross_entropy_with_logits(output, target, reduction='none')
    pt = torch.exp(-bce_loss)  # pt = p if target=1, else 1-p
    
    focal_loss = alpha * (1 - pt) ** gamma * bce_loss
    
    if reduction == 'mean':
        return focal_loss.mean()
    elif reduction == 'sum':
        return focal_loss.sum()
    else:
        return focal_loss
    
    
    
def focal_loss(output, target, alpha=0.25, gamma=2.0, reduction='mean'):
    bce_loss = F.binary_cross_entropy(output, target, reduction='none')
    pt = torch.where(target == 1, output, 1 - output)  # pt = p if target=1, else 1-p

    focal_loss = alpha * (1 - pt) ** gamma * bce_loss
    
    if reduction == 'mean':
        return focal_loss.mean()
    elif reduction == 'sum':
        return focal_loss.sum()
    else:
        return focal_loss