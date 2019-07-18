import torch
from custom_pytorch.losses.dice import DiceCoeff
from custom_pytorch.custom_utils import apply_reduction
class BCEDiceLoss(DiceCoeff):
    def forward(self, input, target, wrt_batch=True, reduction='mean'):

        pos_coeffs = super().__call__(input, target, wrt_batch, reduction='none')
        neg_coeffs = super().__call__(1 - input, 1 - target, wrt_batch, reduction='none')
        bce = - (torch.log2(pos_coeffs) + torch.log2(neg_coeffs))
        if wrt_batch:
            return apply_reduction(bce, reduction)
        return bce


class DiceLoss(BCEDiceLoss):
    def __init__(self, *args, with_bce=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.with_bce = with_bce

    def forward(self, input, target, *args, **kwargs):
        if self.with_bce:
            return super().__call__(input, target, *args, **kwargs)
        return 1 - DiceCoeff.__call__(self, input, target, *args, **kwargs)

