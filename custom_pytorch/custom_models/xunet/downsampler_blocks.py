from .base import _Downsampler
from torch import nn
from custom_pytorch.custom_layers.custom_xception_with_se import SEXceptionBlock, XceptionBlock
from custom_pytorch.custom_utils.compute_layers import compute_needed_layers


class SimpleDownsamplerBlock(_Downsampler):
    def __init__(self, inp_channels, out_channels):
        super().__init__(inp_channels, out_channels)
        self.sequence = nn.Sequential(
            nn.Conv2d(inp_channels, out_channels, 3, padding=1),
            nn.ReLU6())

    def forward(self, input):
        return self.sequence(input)


class XceptionDownsamplerBlock(_Downsampler):
    def __init__(self, inp_channels, out_channels):
        super().__init__(inp_channels, out_channels)
        self.block = XceptionBlock(inp_channels, out_channels,
                                   compute_needed_layers(inp_channels, out_channels))

    def forward(self, input):
        return self.block(input)


class SEXceptionDownsamplerBlock(_Downsampler):
    def __init__(self, inp_channels, out_channels):
        super().__init__(inp_channels, out_channels)
        self.block = SEXceptionBlock(inp_channels, out_channels,
                                     compute_needed_layers(inp_channels, out_channels))

    def forward(self, input):
        return self.block(input)
