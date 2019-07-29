import torch
from torch import nn
from custom_pytorch.custom_utils import apply_reduction


class DiceCoeff(nn.Module):
    def __init__(self, pos_weight=None, threshold=None, activation='sigmoid'):
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

    def _compute(self, input, target, wrt_batch):
        smooth = 1.
        if self.activation is not None:
            input = self.activation(input)
        if self.pos_weight is not None:
            self.pos_weight = self.pos_weight.to(input.device)
            input = input.view(
                input.size()[0], -1) * self.pos_weight.view(1, -1)
            target = target.view(
                target.size()[0], -1) * self.pos_weight.view(1, -1)
        iflat = input.contiguous().view(input.size()[0], -1)
        if self.threshold is not None:
            iflat = (iflat >= self.threshold).float()
        tflat = target.contiguous().view(input.size()[0], -1)
        intersection = (iflat * tflat)
        addition = iflat + tflat
        if wrt_batch:
            intersection = torch.sum(intersection, dim=1)
            addition = torch.sum(addition, dim=1)
        else:
            intersection = intersection.sum()
            addition = addition.sum()
        return ((2. * intersection + smooth) /
                (addition + smooth))

    def forward(self, input, target, wrt_batch=True, reduction='mean'):
        """
        :param input: the input
        :type input: Tensor
        :param target: the target
        :type target: Tensor
        :param wrt_batch: whether to calculate dice coefficient
            for each sample in the batch separately, defaults to True
        :type wrt_batch: bool, optional
        :param reduction: if `wrt_batch` is provided, the provided reduction will be performed,
            only available is `mean`, `sum` or `none`, defaults to `mean`
        :type reduction: str, optional
        """
        coeff = self._compute(input, target, wrt_batch)
        if wrt_batch:
            return apply_reduction(coeff, reduction)
        return coeff
