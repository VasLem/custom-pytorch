from torch import nn
from custom_pytorch.custom_layers.custom_xception_block import SeparableConv2d


class SeparableConv2dReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0,
                 stride=1, use_batchnorm=True, relu_to_use="leaky", **batchnorm_params):
        super().__init__()
        if relu_to_use == 'leaky':
            relu = nn.LeakyReLU(inplace=False)
        elif relu_to_use == '6':
            relu = nn.ReLU6(inplace=False)
        elif relu_to_use == 'random':
            relu = nn.RReLU(inplace=False)
        else:
            relu = nn.ReLU(inplace=False)

        layers = [
            SeparableConv2d(in_channels, out_channels, kernel_size,
                            stride=stride, padding=padding, bias=not (use_batchnorm)),
            relu,
        ]

        if use_batchnorm:
            layers.insert(1, nn.BatchNorm2d(out_channels, **batchnorm_params))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)
