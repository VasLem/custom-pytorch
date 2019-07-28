from custom_pytorch.custom_layers.custom_xception_with_se import SEXceptionBlock
from .base import _OutputBlock
import torch.nn.functional as F
import math
from custom_pytorch.custom_utils.compute_layers import compute_needed_layers

class SEXceptionOutputBlock(_OutputBlock):
    def __init__(self, in_channels, out_channels):
        super().__init__(in_channels, out_channels)
        self.reps = compute_needed_layers(in_channels, out_channels)

        self.block = SEXceptionBlock(in_channels, out_channels, self.reps,
            end_with_relu=False, expand_first=False)
        self.initialize()

    def forward(self, x):
        return self.block(x)
