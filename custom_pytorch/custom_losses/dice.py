import torch
from torch import nn
from custom_pytorch.metrics import DiceCoeff
import torch.nn.functional as F
from custom_pytorch.losses.dice import DiceLoss as _DiceLoss
from custom_pytorch.custom_utils import apply_reduction


__all__ = ['CEDiceLoss', 'DiceLoss', 'BCEAndDiceLoss',
           'BCEAndCEDiceLoss', 'WindowedCEDiceLoss', 'BCEAndWindowedCEDiceLoss']


class DiceLoss(_DiceLoss):
    __name__ = 'dice_loss'

    def __init__(self, with_logits, pos_weight=None, threshold=None):
        activation = None
        if with_logits:
            activation = nn.Sigmoid()
        super().__init__(pos_weight=pos_weight, threshold=threshold, activation=activation)


class WindowedCEDiceLoss(nn.Module):
    __name__ = 'windowed_ce_dice_loss'

    def __init__(self, with_logits=True, red_kernel_size=3, pos_weight=None, threshold=None):
        super().__init__()
        self.activation = None
        if with_logits:
            self.activation = nn.Sigmoid()
        self.dice_coeff = DiceCoeff(
            activation=None, pos_weight=pos_weight, threshold=threshold)
        self.red_kernel_size = red_kernel_size
        self.kernel = torch.ones(
            (1, 1, red_kernel_size, red_kernel_size)) / float(red_kernel_size ** 2)
        self.pad = nn.ConstantPad2d((self.red_kernel_size - 1) // 2, 0)

    def forward(self, input, target, reduction='mean'):
        if self.activation is not None:
            input = self.activation(input)
        kernel = self.kernel.to(input.device)
        input = self.pad(F.conv2d(input,
                                  kernel))
        target = self.pad(F.conv2d(target, kernel))
        pos_coeffs = self.dice_coeff(
            input, target, _no_reduce=True)
        neg_coeffs = self.dice_coeff(
            1 - input, 1 - target, _no_reduce=True)
        loss = - (torch.log2(pos_coeffs + 1e-7) +
                  torch.log2(neg_coeffs + 1e-7))

        return apply_reduction(loss, reduction)


class CEDiceLoss(nn.Module):
    __name__ = 'ce_dice_loss'

    def __init__(self, with_logits=True, pos_weight=None, threshold=None):
        super().__init__()
        self.activation = None
        if with_logits:
            self.activation = nn.Sigmoid()
        self.dice_coeff = DiceCoeff(
            activation=None, pos_weight=pos_weight, threshold=threshold)

    def forward(self, input, target):
        if self.activation is not None:
            input = self.activation(input)
        pos_coeffs = self.dice_coeff(
            input, target)
        neg_coeffs = self.dice_coeff(
            1 - input, 1 - target)
        loss = - (torch.log2(pos_coeffs + 1e-7) +
                  torch.log2(neg_coeffs + 1e-7))
        return loss


class BCEAndDiceLoss(nn.Module):
    __name__ = 'bce_and_dice_loss'

    def __init__(self, with_logits=True, dice_loss_kwargs={}, bce_kwargs={}, _dice_loss_class=None):
        super().__init__()
        self.with_logits = with_logits
        if _dice_loss_class is None:
            _dice_loss_class = DiceLoss
        self.dice_loss = _dice_loss_class(
            with_logits=with_logits, **dice_loss_kwargs)
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
        super().__init__(with_logits, dice_loss_kwargs,
                         bce_kwargs, _dice_loss_class=CEDiceLoss)


class BCEAndWindowedCEDiceLoss(BCEAndDiceLoss):
    __name__ = 'bce_and_windowed_ce_dice_loss'

    def __init__(self, with_logits=True, dice_loss_kwargs={}, bce_kwargs={}):
        super().__init__(with_logits, dice_loss_kwargs,
                         bce_kwargs, _dice_loss_class=WindowedCEDiceLoss)
