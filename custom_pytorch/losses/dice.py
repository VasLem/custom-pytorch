import torch
from torch.autograd import Variable
from torch import nn

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

class DiceCoeff(nn.Module):
    def __init__(self, pos_weight=None, threshold=None):
        """
        :param pos_weight: Positional weights given to each element of the tensor,
            defaults to None
        :type pos_weight: Tensor|np.ndarray, optional
        :param threshold: The threshold to compare dice coeffient,
            defaults to None
        :type threshold: numeric, optional
        """
        super().__init__()
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
        if self.pos_weight is not None:
            self.pos_weight = self.pos_weight.to(input.device)
            input = input.view(input.size()[0], -1) * self.pos_weight.view(1, -1)
            target = target.view(target.size()[0], -1) * self.pos_weight.view(1, -1)
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

        assert intersection.size()[0] == input.size()[0]
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

class DiceLoss(DiceCoeff):
    def forward(self, *args, **kwargs):
        coeff = super().forward(*args, **kwargs)
        return 1 - coeff