from custom_pytorch.custom_layers.custom_xception_with_se import SEXceptionBlock
from .base import _OutputBlock
import torch.nn.functional as F
from torch import nn
import math
from custom_pytorch.custom_utils.compute_layers import compute_needed_layers
from custom_pytorch.custom_layers import SeparableConv2dReLU, Conv2dReLU


class SimpleOutputBlock(_OutputBlock):
    def __init__(self, in_channels, out_channels):
        super().__init__(in_channels, out_channels)
        self.block = nn.Sequential(
            nn.Dropout2d(0.2),
            Conv2dReLU(in_channels, in_channels, 3, padding=1),
            Conv2dReLU(in_channels, out_channels, 3, padding=1),
            Conv2dReLU(out_channels, out_channels, 3, padding=1),
            nn.Conv2d(out_channels, out_channels, 1, padding=0))
        self.initialize()

    def forward(self, x):
        return self.block(x)


class XceptionOutputBlock(_OutputBlock):
    def __init__(self, in_channels, out_channels):
        super().__init__(in_channels, out_channels)
        self.block = nn.Sequential(
            SeparableConv2dReLU(in_channels, in_channels, 1),
            nn.Dropout2d(0.2),
            SeparableConv2dReLU(in_channels, out_channels, 3, padding=1),
            nn.Conv2d(out_channels, out_channels, 3, padding=1))
        self.initialize()

    def forward(self, x):
        return self.block(x)


class SEXceptionOutputBlock(_OutputBlock):
    def __init__(self, in_channels, out_channels):
        super().__init__(in_channels, out_channels)
        self.block = nn.Sequential(
            SeparableConv2dReLU(in_channels, in_channels, 1),
            nn.Dropout2d(0.2),
            SeparableConv2dReLU(in_channels, out_channels, 3, padding=1),
            nn.Conv2d(out_channels, out_channels, 3, padding=1))
        self.initialize()

    def forward(self, x):
        return self.block(x)
