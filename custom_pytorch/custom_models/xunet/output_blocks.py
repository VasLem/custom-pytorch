from custom_pytorch.custom_layers.custom_xception_with_se import SEXceptionBlock
from .base import _OutputBlock
import torch.nn.functional as F
from torch import nn
import math
from custom_pytorch.custom_utils.compute_layers import compute_needed_layers
from custom_pytorch.custom_layers.separable_conv2relu import SeparableConv2dReLU


class SEXceptionOutputBlock(_OutputBlock):
    def __init__(self, in_channels, out_channels):
        super().__init__(in_channels, out_channels)
        self.reps = compute_needed_layers(in_channels, out_channels)

        # self.block = nn.Sequential(SEXceptionBlock(in_channels, out_channels, self.reps,
        #     end_with_relu=True, expand_first=False),
        #     nn.Conv2d(out_channels, out_channels, 3, padding=1))
        self.block = nn.Sequential(
            SeparableConv2dReLU(in_channels, in_channels, 1),
            nn.Dropout2d(0.2),
            SeparableConv2dReLU(in_channels, out_channels, 3, padding=1),
            nn.Conv2d(out_channels, out_channels, 3, padding=1))
        self.initialize()

    def forward(self, x):
        return self.block(x)
