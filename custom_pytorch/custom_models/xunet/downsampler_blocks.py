from .base import _Downsampler
from torch import nn
from custom_pytorch.custom_layers.custom_xception_with_se import SEXceptionBlock
from custom_pytorch.custom_utils.compute_layers import compute_needed_layers
from custom_pytorch.custom_layers.separable_conv2relu import SeparableConv2dReLU

class SimpleDownsamplerBlock(_Downsampler):
    def __init__(self, inp_channels, out_channels):
        super().__init__(inp_channels, out_channels)
        self.sequence = nn.Sequential(
            nn.Conv2d(inp_channels, out_channels, 3, padding=1),
            nn.ReLU6())
        self.initialize()

    def forward(self, input):
        return self.sequence(input)

class SEXceptionDownsamplerBlock(_Downsampler):
    def __init__(self, inp_channels, out_channels):
        super().__init__(inp_channels, out_channels)
        self.sequence = SEXceptionBlock(inp_channels, out_channels,
                                        compute_needed_layers(inp_channels, out_channels))
        # self.sequence = nn.Sequential(
        #     SeparableConv2dReLU(inp_channels, out_channels, 3, padding=1))
        self.initialize()

    def forward(self, input):
        return self.sequence(input)
