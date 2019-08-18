import torch
from torch import nn
import numpy as np

class DiceCoeff(nn.Module):
    def __init__(self, pos_weight=None, threshold=None, activation=None):
        """
        :param pos_weight: Positional weights given to each element of the tensor,
            defaults to None
        :type pos_weight: Tensor|np.ndarray, optional
        :param threshold: The threshold to compare dice coeffient,
            defaults to None
        :type threshold: numeric, optional
        """
        super().__init__()
        self.activation = activation
        if self.activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        if pos_weight is not None:
            try:
                pos_weight.size()
            except TypeError:
                pos_weight = torch.from_numpy(pos_weight)
            except AttributeError:
                raise ValueError(
                    "Supplied pos_weight is neither"
                    " a Tensor nor a numpy array")
            pos_weight = pos_weight.detach()
        self.pos_weight = pos_weight
        if threshold is not None:
            try:
                threshold = float(threshold)
            except TypeError:
                raise ValueError("Supplied threshold is not numeric")
        self.threshold = threshold

    def _compute(self, input, target):
        smooth = 1.
        input = input.contiguous()
        target = target.contiguous()
        if self.activation is not None:
            input = self.activation(input)
        iflat = input.view(input.size()[0], -1)
        # assert np.all(iflat.cpu().data.numpy() > 0)
        tflat = target.view(input.size()[0], -1)
        # assert np.all(tflat.cpu().data.numpy() >= 0), np.unique(tflat.cpu().data.numpy())
        if self.pos_weight is not None:
            self.pos_weight = self.pos_weight.to(input.device)
            iflat = iflat * self.pos_weight.view(1, -1)
            tflat = tflat * self.pos_weight.view(1, -1)
        if self.threshold is not None:
            iflat = (iflat >= self.threshold).float()
        intersection = (iflat * tflat)
        addition = iflat + tflat
        intersection = intersection.sum()
        addition = addition.sum()
        return ((2. * intersection + smooth) /
                (addition + smooth))

    def forward(self, input, target):
        """
        :param input: the input
        :type input: Tensor
        :param target: the target
        :type target: Tensor
        """
        coeff = self._compute(input, target)
        return coeff
