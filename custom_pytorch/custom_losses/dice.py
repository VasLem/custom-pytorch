import torch
from torch import nn
import numpy as np
from custom_pytorch.metrics import DiceCoeff
import torch.nn.functional as F
from custom_pytorch.losses.dice import DiceLoss as _DiceLoss
from custom_pytorch.custom_utils import apply_reduction


__all__ = ['CEDiceLoss', 'DiceLoss', 'BCEAndDiceLoss', 'BCEAndCEDiceLoss', 'DilatedCEDiceLoss']


class DiceLoss(_DiceLoss):
    __name__ = 'dice_loss'


class DilatedCEDiceLoss(nn.Module):
    __name__ = 'dilated_ce_dice_loss'

    def __init__(self, *args, red_kernel_size=3, **kwargs):
        super().__init__()
        self.dice_coeff = DiceCoeff(*args, **kwargs)
        self.red_kernel_size = red_kernel_size
        self.kernel = torch.ones(
            (1, 1, red_kernel_size, red_kernel_size)) * 1/(red_kernel_size**2)
        self.pad = nn.ConstantPad2d((self.red_kernel_size - 1) // 2, 1)

    def forward(self, input, target, reduction='mean'):
        self.kernel = self.kernel.to(input.device)
        pos_coeffs = self.dice_coeff(
            input, target, wrt_batch=True, reduction='none', _no_reduce=True)
        pos_coeffs = self.pad(F.conv2d(pos_coeffs,
                                       self.kernel))
        neg_coeffs = self.dice_coeff(
            1 - self.dice_coeff.activation(input),
            1 - target, wrt_batch=True, reduction='none', _no_reduce=True,
            do_activation=False)
        neg_coeffs = self.pad(F.conv2d(neg_coeffs,
                                       self.kernel))
        # remove the outer bounds from the play -> a necessary drawback of this loss
        mask = (self.pad(F.conv2d(target, self.kernel)) > 0).float() - target
        neg_coeffs[mask] = 1
        pos_coeffs[mask] = 1
        loss = - (torch.log2(pos_coeffs + 1e-7) +
                  torch.log2(neg_coeffs + 1e-7))

        return apply_reduction(loss, reduction)


class CEDiceLoss(nn.Module):
    __name__ = 'dilated_ce_dice_loss'

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.dice_coeff = DiceCoeff(*args, **kwargs)

    def forward(self, input, target, wrt_batch=True, reduction='mean'):
        pos_coeffs = self.dice_coeff(
            input, target, wrt_batch=wrt_batch, reduction='none')
        neg_coeffs = self.dice_coeff(
            1 - self.dice_coeff.activation(input),
            1 - target, wrt_batch=wrt_batch, reduction='none',
            do_activation=False)
        loss = - (torch.log2(pos_coeffs + 1e-7) +
                  torch.log2(neg_coeffs + 1e-7))
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
