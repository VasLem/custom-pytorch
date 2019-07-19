import torch
from torch import nn
from custom_pytorch.metrics import DiceCoeff
from custom_pytorch.losses.dice import DiceLoss as _DiceLoss
from custom_pytorch.custom_utils import apply_reduction
class BCEDiceLoss(DiceCoeff):
    def forward(self, input, target, wrt_batch=True, reduction='mean'):

        pos_coeffs = super().__call__(input, target, wrt_batch, reduction='none')
        neg_coeffs = super().__call__(1 - input, 1 - target, wrt_batch, reduction='none')
        bce = - (torch.log2(pos_coeffs) + torch.log2(neg_coeffs))
        if wrt_batch:
            return apply_reduction(bce, reduction)
        return bce


class DiceLoss(nn.Module):
    def __init__(self, *args, with_bce=False, **kwargs):
        super().__init__()
        if with_bce:
            self.loss = BCEDiceLoss(*args, **kwargs)
        else:
            self.loss = _DiceLoss(*args, **kwargs)

    def forward(self, input, target, *args, **kwargs):
        return self.loss(input, target, *args, **kwargs)

class BCEAndDiceLoss(nn.Module):
    def __init__(self, with_logits=True, dice_loss_kwargs={}, bce_kwargs={}):
        super().__init__()
        self.dice_loss = DiceLoss(**dice_loss_kwargs)
        self.with_logits = with_logits
        if self.with_logits:
            self.bce_loss = torch.nn.BCEWithLogitsLoss(**bce_kwargs)
        else:
            self.bce_loss = torch.nn.BCELoss(**bce_kwargs)

    def forward(self, inputs, targets):
        if self.with_logits:
            return self.dice_loss(torch.sigmoid(inputs), targets) + self.bce_loss(inputs, targets)
        else:
            return self.dice_loss(inputs, targets) + self.bce_loss(inputs, targets)