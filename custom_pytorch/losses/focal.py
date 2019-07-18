import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class FocalLoss(nn.Module):
    """It is unstable due to BCELoss
    """
    def __init__(self, alpha=0.2, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        if logits:
            self.inner_loss_func = nn.BCEWithLogitsLoss(reduction='none')
        else:
            self.inner_loss_func = nn.BCELoss(reduction='none')
        self.reduce = reduce

    def forward(self, inputs, targets):
        BCE_loss = self.inner_loss_func(inputs, targets)
        assert np.all(BCE_loss.cpu().data.numpy() >= 0), (np.unique(targets.cpu().data.numpy()),
                                                          np.min(inputs.cpu().data.numpy()),
                                                          np.max(inputs.cpu().data.numpy()))
        pt = torch.exp(-BCE_loss)
        assert np.all(1 - pt.cpu().data.numpy())**self.gamma <= 1
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        if self.reduce:
            m = torch.mean(F_loss)
            return m
        return F_loss