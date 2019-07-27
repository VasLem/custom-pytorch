from .base import _Downsampler
from torch import nn
from custom_pytorch.custom_layers.custom_xception_with_se import SEXceptionBlock

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
        self.sequence = SEXceptionBlock(inp_channels, out_channels, 1)
        self.initialize()

    def forward(self, input):
        return self.sequence(input)
