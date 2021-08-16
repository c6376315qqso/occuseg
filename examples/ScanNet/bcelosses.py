

import torch
import torch.nn as nn


def WeightedBCELoss(output, target, weight):
    output = torch.clamp(output,min=1e-8,max=1-1e-8)  
    loss =  weight[1] * (target * torch.log(output)) + weight[0] * ((1 - target) * torch.log(1 - output))
    loss = torch.neg(torch.mean(loss))
    return loss


class BCEFocalLoss(torch.nn.Module):
    """
    二分类的Focalloss alpha 固定
    alpha: postive weight 
    """
    def __init__(self, gamma=2, alpha=0.25, reduction='elementwise_mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
 
    def forward(self, _input, target):
        pt = torch.sigmoid(_input)
        pt = torch.clamp(pt,min=1e-8,max=1-1e-8)
        alpha = self.alpha
        loss = - alpha * (1 - pt) ** self.gamma * target * torch.log(pt) - \
               (1 - alpha) * pt ** self.gamma * (1 - target) * torch.log(1 - pt)
        if self.reduction == 'elementwise_mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss

