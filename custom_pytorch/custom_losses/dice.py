import torch
from torch import nn
from custom_pytorch.metrics import DiceCoeff
import torch.nn.functional as F
from custom_pytorch.losses.dice import DiceLoss as _DiceLoss


__all__ = ['CEDiceLoss', 'DiceLoss', 'BCEAndDiceLoss',
           'BCEAndCEDiceLoss', 'WindowedCEDiceLoss', 'BCEAndWindowedCEDiceLoss',
           'MultiWindowedCEDiceLoss', 'BCEAndMultiWindowedCEDiceLoss',
           'MultiWindowedDiceLoss', 'WindowedDiceLoss', 'MultiWindowedBCELoss', 'WindowedBCELoss',
           'BCEAndMultiWindowedDiceLoss']

# Inherit from Function


class NoBackpropConv2d(torch.autograd.Function):

    # Note that both forward and backward are @staticmethods
    @staticmethod
    def forward(ctx, input, kernel, pad_value=0):
        input = F.conv2d(input, kernel)
        p = (kernel.size()[-1] - 1)//2
        return F.pad(input, (p, p, p, p), value=pad_value)

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        return grad_output, None


nobackprop_conv2d = NoBackpropConv2d.apply


class DiceLoss(_DiceLoss):
    __name__ = 'dice_loss'

    def __init__(self, with_logits, pos_weight=None, threshold=None):
        activation = None
        if with_logits:
            activation = nn.Sigmoid()
        super().__init__(pos_weight=pos_weight,
                         threshold=threshold, activation=activation)


class _Windowed(nn.Module):
    def __init__(self, *args, red_kernel_size=3, **kwargs):
        super().__init__()
        self.red_kernel_size = red_kernel_size
        self.kernel = torch.ones(
            (1, 1, red_kernel_size, red_kernel_size)) / float(red_kernel_size ** 2)

    def forward(self, input, target, *args, **kwargs):
        kernel = self.kernel.to(input.device)
        input = nobackprop_conv2d(input, kernel)
        target = nobackprop_conv2d(target, kernel)
        return input, target


class _MultiWindowed(nn.Module):

    def __init__(self, windowed_class, win_num=3, **kwargs):
        super().__init__()
        self.windowed_components = nn.ModuleList(
            [windowed_class(red_kernel_size=(2*(w + 1) + 1), **kwargs)
             for w in range(win_num)])

    def forward(self, input, target, **kwargs):
        loss = None
        for component in self.windowed_components:
            if loss is None:
                loss = component(input, target, **kwargs)
            else:
                loss += component(input, target, **kwargs)
        return loss / len(self.windowed_components)


class WindowedCEDiceLoss(_Windowed):
    __name__ = 'windowed_ce_dice_loss'

    def __init__(self, with_logits=True, red_kernel_size=3, pos_weight=None, threshold=None):
        super().__init__(red_kernel_size=3)
        self.activation = None
        if with_logits:
            self.activation = nn.Sigmoid()
        self.dice_coeff = DiceCoeff(
            activation=None, pos_weight=pos_weight, threshold=threshold)

    def forward(self, input, target, reduction='mean'):
        if self.activation is not None:
            input = self.activation(input)
        input, target = super().forward(input, target)

        pos_coeffs = self.dice_coeff(
            input, target)
        neg_coeffs = self.dice_coeff(
            1 - input, 1 - target)
        loss = - (torch.log2(pos_coeffs + 1e-7) +
                  torch.log2(neg_coeffs + 1e-7))
        return loss


class WindowedDiceLoss(_Windowed):
    __name__ = 'windowed_dice_loss'

    def __init__(self, with_logits=True, red_kernel_size=3, pos_weight=None, threshold=None):
        super().__init__(red_kernel_size=3)
        self.activation = None
        if with_logits:
            self.activation = nn.Sigmoid()
        self.dice_coeff = DiceCoeff(
            activation=None, pos_weight=pos_weight, threshold=threshold)

    def forward(self, input, target, reduction='mean'):
        if self.activation is not None:
            input = self.activation(input)
        input, target = super().forward(input, target)

        return 1 - self.dice_coeff(
            input, target)


class WindowedBCELoss(_Windowed):
    __name__ = 'windowed_bce_loss'

    def __init__(self, with_logits=True, red_kernel_size=3, **bce_kwargs):
        super().__init__(red_kernel_size=3)
        if with_logits:
            self.bce_loss = torch.nn.BCEWithLogitsLoss(**bce_kwargs)
        else:
            self.bce_loss = torch.nn.BCELoss(**bce_kwargs)

    def forward(self, input, target, **kwargs):
        input, target = super().forward(input, target)
        return self.bce_loss(
            input, target, **kwargs)


class MultiWindowedDiceLoss(_MultiWindowed):
    __name__ = 'multi_windowed_dice_loss'

    def __init__(self, with_logits=True, win_num=3, pos_weight=None, threshold=None):
        super().__init__(WindowedDiceLoss, win_num=win_num, with_logits=with_logits,
                         pos_weight=pos_weight, threshold=threshold)


class MultiWindowedCEDiceLoss(_MultiWindowed):
    __name__ = 'multi_windowed_ce_dice_loss'

    def __init__(self, with_logits=True, win_num=3, pos_weight=None, threshold=None):
        super().__init__(WindowedCEDiceLoss, win_num=win_num, with_logits=with_logits,
                         pos_weight=pos_weight, threshold=threshold)


class MultiWindowedBCELoss(_MultiWindowed):
    __name__ = 'multi_windowed_bce_loss'

    def __init__(self, with_logits=True, win_num=3, **bce_kwargs):
        super().__init__(WindowedBCELoss, win_num=win_num, with_logits=with_logits,
                         **bce_kwargs)


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
        return self.dice_loss(inputs, targets, **dice_loss_kwargs) + \
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


class BCEAndMultiWindowedCEDiceLoss(BCEAndDiceLoss):
    __name__ = 'bce_and_multiwindowed_ce_dice_loss'

    def __init__(self, with_logits=True, dice_loss_kwargs={}, bce_kwargs={}):
        super().__init__(with_logits, dice_loss_kwargs,
                         bce_kwargs, _dice_loss_class=MultiWindowedCEDiceLoss)


class BCEAndMultiWindowedDiceLoss(BCEAndDiceLoss):
    __name__ = 'bce_and_multiwindowed_dice_loss'

    def __init__(self, with_logits=True, dice_loss_kwargs={}, bce_kwargs={}):
        super().__init__(with_logits, dice_loss_kwargs,
                         bce_kwargs, _dice_loss_class=MultiWindowedDiceLoss)
