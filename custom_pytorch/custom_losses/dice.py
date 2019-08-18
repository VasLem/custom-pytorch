import torch
from torch import nn
from custom_pytorch.metrics import DiceCoeff
import torch.nn.functional as F
from custom_pytorch.losses.dice import DiceLoss as _DiceLoss
from torchnet.meter import AverageValueMeter

__all__ = ['CEDiceLoss', 'DiceLoss', 'BCEAndDiceLoss',
           'BCEAndCEDiceLoss', 'WindowedCEDiceLoss', 'BCEAndWindowedCEDiceLoss',
           'MultiWindowedCEDiceLoss', 'BCEAndMultiWindowedCEDiceLoss',
           'MultiWindowedDiceLoss', 'WindowedDiceLoss', 'MultiWindowedBCELoss', 'WindowedBCELoss',
           'BCEAndMultiWindowedDiceLoss']


class NoBackpropConv2d(torch.autograd.Function):
    """Conv2D with no backpropagation, is invisible during backward pass

    """
    @staticmethod
    def forward(ctx, input, kernel, pad_value=0):
        input = F.conv2d(input, kernel)
        p = (kernel.size()[-1] - 1)//2
        return F.pad(input, (p, p, p, p), value=pad_value)

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


nobackprop_conv2d = NoBackpropConv2d.apply


class _Logger:

    def update_logs(self, logs, loss):
        if logs is not None:
            if self.__name__ not in logs:
                logs[self.__name__] = AverageValueMeter()
            logs[self.__name__].add(loss.cpu().data.numpy())


class DiceLoss(_DiceLoss, _Logger):
    __name__ = 'dice_loss'

    def __init__(self, with_logits, pos_weight=None, threshold=None):
        activation = None
        if with_logits:
            activation = nn.Sigmoid()
        super().__init__(pos_weight=pos_weight,
                         threshold=threshold, activation=activation)

    def forward(self, *args, logs=None, **kwargs):
        loss = super().forward(*args, **kwargs)
        self.update_logs(logs, loss)
        return loss


class CEDiceLoss(nn.Module, _Logger):
    __name__ = 'ce_dice_loss'

    def __init__(self, with_logits=True, pos_weight=None, threshold=None):
        super().__init__()
        self.activation = None
        if with_logits:
            self.activation = nn.Sigmoid()
        self.dice_coeff = DiceCoeff(
            activation=None, pos_weight=pos_weight, threshold=threshold)

    def forward(self, input, target, logs=None):
        if self.activation is not None:
            input = self.activation(input)
        pos_coeffs = self.dice_coeff(
            input, target)
        neg_coeffs = self.dice_coeff(
            1 - input, 1 - target)
        loss = - (torch.log2(pos_coeffs + 1e-7) +
                  torch.log2(neg_coeffs + 1e-7))
        self.update_logs(logs, loss)
        return loss


class BCELoss(nn.Module, _Logger):
    __name__ = 'bce_loss'

    def __init__(self, with_logits=True, **kwargs):
        super().__init__()
        self.with_logits = with_logits
        if self.with_logits:
            self.bce_loss = torch.nn.BCEWithLogitsLoss(**kwargs)
        else:
            self.bce_loss = torch.nn.BCELoss(**kwargs)

    def forward(self, inputs, targets, logs=None):
        loss = self.bce_loss(inputs, targets)
        self.update_logs(logs, loss)
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
        self.bce_loss = BCELoss(with_logits=self.with_logits, **bce_kwargs)

    def forward(self, inputs, targets, logs=None, dice_loss_kwargs={}, bce_kwargs={}):
        return self.dice_loss(inputs, targets, logs=logs, **dice_loss_kwargs) + \
            self.bce_loss(inputs, targets, logs=logs, **bce_kwargs)


class BCEAndCEDiceLoss(BCEAndDiceLoss):
    __name__ = 'bce_and_ce_dice_loss'

    def __init__(self, with_logits=True, dice_loss_kwargs={}, bce_kwargs={}):
        super().__init__(with_logits, dice_loss_kwargs,
                         bce_kwargs, _dice_loss_class=CEDiceLoss)


class _Windowed(nn.Module):
    def __init__(self, loss_class, with_logits, red_kernel_size=3):
        super().__init__()
        self.red_kernel_size = red_kernel_size
        self.loss_class = loss_class
        self.with_logits = with_logits
        self.activation = None
        if self.with_logits:
            self.activation = nn.Sigmoid()
        self.kernel = torch.ones(
            (1, 1, red_kernel_size, red_kernel_size)) / float(red_kernel_size ** 2)

    def forward(self, input, target, *args, logs=None, **kwargs):
        kernel = self.kernel.to(input.device)
        if self.with_logits:
            input = self.activation(input)
        input = nobackprop_conv2d(input, kernel)
        target = nobackprop_conv2d(target, kernel)
        return self.loss_class(input, target, *args, logs=logs, **kwargs)


class WindowedCEDiceLoss(_Windowed):
    __name__ = 'windowed_ce_dice_loss'

    def __init__(self, with_logits=True, red_kernel_size=3, pos_weight=None, threshold=None):
        loss_class = CEDiceLoss(with_logits=False,
                                pos_weight=pos_weight, threshold=threshold)
        loss_class.__name__ = f'ce_dice_loss, window: {red_kernel_size}'
        super().__init__(with_logits=with_logits,
                         loss_class=loss_class,
                         red_kernel_size=red_kernel_size)


class WindowedDiceLoss(_Windowed):
    __name__ = 'windowed_dice_loss'

    def __init__(self, with_logits=True, red_kernel_size=3, pos_weight=None, threshold=None):
        loss_class = DiceLoss(with_logits=False,
                              pos_weight=pos_weight, threshold=threshold)
        loss_class.__name__ = f'dice_loss, window: {red_kernel_size}'
        super().__init__(with_logits=with_logits,
                         loss_class=loss_class,
                         red_kernel_size=red_kernel_size)


class WindowedBCELoss(_Windowed):
    __name__ = 'windowed_bce_loss'

    def __init__(self, with_logits=True, red_kernel_size=3, pos_weight=None, threshold=None):
        loss_class = BCELoss(with_logits=with_logits,
                             pos_weight=pos_weight, threshold=threshold)
        loss_class.__name__ = f'bce_loss, window: {red_kernel_size}'
        super().__init__(loss_class=loss_class,
                         red_kernel_size=red_kernel_size)


class _MultiWindowed(nn.Module):

    def __init__(self, windowed_class, win_num=3, **kwargs):
        super().__init__()
        self.windowed_components = nn.ModuleList(
            [windowed_class(red_kernel_size=(2*(w) + 1), **kwargs)
             for w in range(win_num)])

    def forward(self, input, target, *args, logs=None, **kwargs):
        loss = None
        for component in self.windowed_components:
            comp_l = component(input, target, *args, logs=logs, **kwargs)
            if loss is None:
                loss = comp_l
            else:
                loss += comp_l
        return loss


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