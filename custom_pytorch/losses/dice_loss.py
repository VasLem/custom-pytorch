import torch
from torch.autograd import Function, Variable
import torch.nn.functional as F
from torch import nn as nn
from torch.autograd import Variable
from torch.nn import MSELoss, SmoothL1Loss, L1Loss

def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.view(C, -1)


class GeneralizedDiceLoss(nn.Module):
    """Computes Generalized Dice Loss (GDL) as described in https://arxiv.org/pdf/1707.03237.pdf
    """

    def __init__(self, epsilon=1e-5, weight=None, ignore_index=None, sigmoid_normalization=True):
        super(GeneralizedDiceLoss, self).__init__()
        self.epsilon = epsilon
        self.register_buffer('weight', weight)
        self.ignore_index = ignore_index
        if sigmoid_normalization:
            self.normalization = nn.Sigmoid()
        else:
            self.normalization = nn.Softmax(dim=1)

    def forward(self, input, target):
        # get probabilities from logits
        input = self.normalization(input)

        assert input.size() == target.size(), "'input' and 'target' must have the same shape"

        # mask ignore_index if present
        if self.ignore_index is not None:
            mask = target.clone().ne_(self.ignore_index)
            mask.requires_grad = False

            input = input * mask
            target = target * mask

        input = flatten(input)
        target = flatten(target)

        target = target.float()
        target_sum = target.sum(-1)
        class_weights = Variable(1. / (target_sum * target_sum).clamp(min=self.epsilon), requires_grad=False)

        intersect = (input * target).sum(-1) * class_weights
        if self.weight is not None:
            weight = Variable(self.weight, requires_grad=False)
            intersect = weight * intersect
        intersect = intersect.sum()

        denominator = ((input + target).sum(-1) * class_weights).sum()

        return 1. - 2. * intersect / denominator.clamp(min=self.epsilon)


def dice_coeff(input, target, pos_weight=None, threshold=None):
    smooth = 1.
    if pos_weight is not None:
        input = input.view(input.size()[0], -1) * pos_weight.view(1, -1)
        target = target.view(target.size()[0], -1) * pos_weight.view(1, -1)
    iflat = input.contiguous().view(-1)
    if threshold is not None:
        iflat = (iflat >= threshold).float()
    tflat = target.contiguous().view(-1)

    intersection = (iflat * tflat).sum()

    return ((2. * intersection + smooth) /
              (iflat.sum() + tflat.sum() + smooth))

def average_dice_coeff(inputs, targets, pos_weight=None, threshold=None):
    coeff = 0
    for input, target in zip(inputs, targets):
        coeff += dice_coeff(input, target, pos_weight, threshold)
    return coeff / len(inputs)

def dice_loss(input, target, pos_weight=None, threshold=None):
    return 1 - dice_coeff(input, target, pos_weight, threshold)

def average_dice_loss(inputs, targets, pos_weight=None, threshold=None):
    loss = 0
    assert len(inputs) > 0
    for inp, target in zip(inputs, targets):
        loss += dice_loss(inp, target, pos_weight, threshold)
    return loss / len(inputs)

def cross_entropy_dice_loss(input, target, pos_weight=None, threshold=None):
    smooth = 1.
    pos_coeff = dice_coeff(input, target, pos_weight, threshold)
    neg_coeff = dice_coeff(1 - input, 1 - target, pos_weight, threshold)
    return - (torch.log(pos_coeff) + torch.log(neg_coeff))

def where(cond, x_1, x_2):
    return (cond * x_1) + ((1-cond) * x_2)

def _stable_sigmoid(input):
    return where(input < 0, input, 0) - torch.log(torch.exp(-torch.abs(input)) + 1)

def cross_entropy_dice_loss_with_logits(input, target, pos_weight=None, threshold=None):
    pos_coeff = dice_coeff(torch.sigmoid(input), target, pos_weight, threshold)
    neg_coeff = dice_coeff(1 - torch.sigmoid(input), 1 - target, pos_weight, threshold)
    l = - (torch.log(pos_coeff) +
           torch.log(neg_coeff))
    # l = - torch.log(pos_coeff)
    return l

def average_cross_entropy_dice_loss(inputs, targets, pos_weight=None, threshold=None):
    loss = 0
    for inp, target in zip(inputs, targets):
        loss += cross_entropy_dice_loss(inp, target, pos_weight, threshold)
    return loss / len(inputs)

def average_cross_entropy_dice_loss_with_logits(inputs, targets, pos_weight=None, threshold=None):
    loss = 0
    for inp, target in zip(inputs, targets):
        loss += cross_entropy_dice_loss_with_logits(inp, target, pos_weight, threshold)
    return loss / len(inputs)