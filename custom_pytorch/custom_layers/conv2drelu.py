from torch import nn


class Conv2dReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, padding=0,
                 stride=1, use_batchnorm=True, relu_to_use="relu", **batchnorm_params):
        super().__init__()
        if relu_to_use == 'relu':
            relu = nn.ReLU(inplace=False)
        if relu_to_use == 'leaky':
            relu = nn.LeakyReLU(inplace=False)
        elif relu_to_use == '6':
            relu = nn.ReLU6(inplace=False)
        elif relu_to_use == 'random':
            relu = nn.RReLU(inplace=False)
        elif relu_to_use == 'none':
            relu = None
        else:
            relu = nn.ReLU(inplace=False)

        layers = [nn.Conv2d(in_channels, out_channels, kernel_size,
                            stride=stride, padding=padding, bias=not use_batchnorm)]

        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels, **batchnorm_params))
        if relu is not None:
            layers.append(relu)

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)
