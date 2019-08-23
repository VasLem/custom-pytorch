from .base import _DecoderBlock
from torch import nn
import torch.nn.functional as F
from custom_pytorch.custom_layers.custom_xception_with_se import SEXceptionBlock
from custom_pytorch.custom_layers import XceptionBlock
from custom_pytorch.custom_utils.compute_layers import compute_needed_layers
from custom_pytorch.custom_layers import Conv2dReLU



class UnetDecoderBlock(_DecoderBlock):
    def __init__(self, inp_channels, out_channels, scale_ratio, use_batchnorm=True):
        super().__init__(inp_channels, out_channels, scale_ratio)
        self.block = nn.Sequential(
            Conv2dReLU(inp_channels, out_channels, kernel_size=3,
                       padding=1, use_batchnorm=use_batchnorm),
            Conv2dReLU(out_channels, out_channels, kernel_size=3,
                       padding=1, use_batchnorm=use_batchnorm),
        )

    def forward(self, x):
        # x, skip = x
        x = F.interpolate(x, scale_factor=self.scale_ratio, mode='nearest')
        # if skip is not None:
        #     x = torch.cat([x, skip], dim=1)
        x = self.block(x)
        return x


class SimpleDecoderBlock(_DecoderBlock):
    def __init__(self, in_channels, out_channels, scale_ratio, *args, **kwargs):
        super().__init__(in_channels, out_channels, scale_ratio)
        self.block = nn.Sequential(
            Conv2dReLU(in_channels, out_channels, kernel_size=3, padding=1),
            Conv2dReLU(out_channels, out_channels, kernel_size=1))

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_ratio, mode='nearest')
        x = self.block(x)
        return x

class XceptionDecoderBlock(_DecoderBlock):
    def __init__(self, in_channels, out_channels, scale_ratio, *args, **kwargs):
        super().__init__(in_channels, out_channels, scale_ratio)
        self.reps = compute_needed_layers(in_channels, out_channels)
        self.block = XceptionBlock(in_channels, out_channels,
                                   self.reps)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_ratio, mode='nearest')
        x = self.block(x)
        return x


class SEXceptionDecoderBlock(_DecoderBlock):
    def __init__(self, in_channels, out_channels, scale_ratio, *args, **kwargs):
        super().__init__(in_channels, out_channels, scale_ratio)
        self.reps = compute_needed_layers(in_channels, out_channels)
        self.block = SEXceptionBlock(in_channels, out_channels,
                                     self.reps)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_ratio, mode='nearest')
        x = self.block(x)
        return x
