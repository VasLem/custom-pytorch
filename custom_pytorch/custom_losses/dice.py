import torch
from torch import nn
from custom_pytorch.metrics import DiceCoeff
from custom_pytorch.losses.dice import DiceLoss as _DiceLoss
from custom_pytorch.custom_utils import apply_reduction


__all__ = ['CEDiceLoss', 'DiceLoss', 'BCEAndDiceLoss', 'BCEAndCEDiceLoss']


class DiceLoss(_DiceLoss):
    __name__ = 'dice_loss'


class CEDiceLoss(nn.Module):
    __name__ = 'ce_dice_loss'

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.dice_coeff = DiceCoeff(*args, **kwargs)

    def forward(self, input, target, wrt_batch=True, reduction='mean'):
        pos_coeffs = self.dice_coeff(
            input, target, wrt_batch, reduction='none')
        neg_coeffs = self.dice_coeff(
            1 - input, 1 - target, wrt_batch, reduction='none')
        loss = - (pos_coeffs * torch.log2(pos_coeffs) +
                  neg_coeffs * torch.log2(neg_coeffs))
        if wrt_batch:
            return apply_reduction(loss, reduction)
        return loss


class BCEAndDiceLoss(nn.Module):
    __name__ = 'bce_and_dice_loss'

    def __init__(self, with_logits=True, dice_loss_kwargs={}, bce_kwargs={}):
        super().__init__()
        self.with_logits = with_logits
        if with_logits:
            dice_loss_kwargs['activation'] = 'sigmoid'
        self.dice_loss = DiceLoss(**dice_loss_kwargs)
        if self.with_logits:
            self.bce_loss = torch.nn.BCEWithLogitsLoss(**bce_kwargs)
        else:
            self.bce_loss = torch.nn.BCELoss(**bce_kwargs)

    def forward(self, inputs, targets, dice_loss_kwargs={}, bce_kwargs={}):
        return self.dice_loss(inputs, targets, **dice_loss_kwargs) +\
            self.bce_loss(inputs, targets, **bce_kwargs)


class BCEAndCEDiceLoss(BCEAndDiceLoss):
    __name__ = 'bce_and_ce_dice_loss'

    def __init__(self, with_logits=True, dice_loss_kwargs={}, bce_kwargs={}):
        super().__init__(with_logits, dice_loss_kwargs, bce_kwargs)
        self.dice_loss = CEDiceLoss(**dice_loss_kwargs)
