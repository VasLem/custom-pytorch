import numpy as np
from torch import nn
import torch
import torch.nn.functional as F


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=0, dilation=1, bias=False):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size,
                               stride, padding, dilation, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class XceptionBlock(nn.Module):
    def __init__(self, in_filters, out_filters, reps, strides=1,
                 start_with_relu=False, end_with_relu=True):
        super().__init__()
        reps = max(1, reps)
        if out_filters != in_filters or strides != 1:
            self.skip = nn.Conv2d(in_filters, out_filters,
                                  1, stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip = None

        self.relu = nn.ReLU(inplace=False)
        rep = []
        self.in_filters = in_filters
        self.out_filters = out_filters
        if out_filters > in_filters:
            filters_hierarchy = np.linspace(in_filters, out_filters, reps + 1).astype(int)
        else:
            filters_hierarchy = np.linspace(out_filters, in_filters, reps + 1).astype(int)[::-1]
        in_filters_group = filters_hierarchy[:-1].tolist()
        out_filters_group = filters_hierarchy[1:].tolist()
        # print(in_filters, out_filters, reps, filters_hierarchy)


        for in_filt, out_filt in zip(in_filters_group, out_filters_group):
            rep.append(SeparableConv2d(in_filt, out_filt,
                                       3, stride=1, padding=1, bias=False))
            rep.append(self.relu)
            rep.append(nn.BatchNorm2d(out_filt))


        if start_with_relu:
            rep = rep[:-2]
            rep = [nn.ReLU(inplace=False)] + rep
        elif not end_with_relu:
            rep = rep[:-2]

        if strides != 1:
            rep.append(nn.MaxPool2d(3, strides, 1))
        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp
        # print(self.in_filters, self.out_filters, x.size(), skip.size())
        x += skip
        return x


class DownSamplingBlock(nn.Module):
    def __init__(self, n_channels, depth):
        super().__init__()
        self.xception_block = XceptionBlock(n_channels * 2 ** depth,
                                            n_channels * 2 ** (depth + 1),
                                            strides=2, reps=depth)
    def forward(self, image):
        return self.xception_block(image)

class UpSamplingBlock(nn.Module):
    def __init__(self, n_channels, depth, inv_depth):
        super().__init__()
        self.upsampler = nn.UpsamplingNearest2d(scale_factor=2)
        if depth != 0:
            self.xception_block = XceptionBlock(
                n_channels * 2 ** (inv_depth) * 2,
                n_channels * 2 ** max(0, (inv_depth - 1)), reps=4, start_with_relu=False)
        else:
            self.xception_block = XceptionBlock(
                n_channels * 2 ** (inv_depth),
                n_channels * 2 ** max(0, (inv_depth - 1)), reps=4, start_with_relu=False)
    def forward(self, image, previous_output=None):
        if previous_output is not None:
            previous_output = self.upsampler(previous_output)
            image = torch.cat((image, previous_output), dim=1)
        return self.xception_block(image)


class SamplingSegmentationV2(nn.Module):
    def __init__(self, n_channels, n_categories, depth, resolution=1):
        super().__init__()
        self.n_categories = n_categories
        self.n_channels = n_channels
        self.depth = depth
        self.up_sampling_blocks = nn.ModuleList([UpSamplingBlock(n_channels, d, self.depth - d)
                                     for d in range(self.depth + 1)])
        self.down_sampling_blocks = nn.ModuleList([DownSamplingBlock(n_channels, d)
                                     for d in range(self.depth)])
        self.upsamplers = nn.ModuleList([nn.UpsamplingNearest2d(
            scale_factor=2 ** d) for d in range(self.depth + 1)])
        self.dropout = nn.Dropout2d(p=0.2, inplace=False)
        self.final_layer1 = XceptionBlock(
            self.n_channels * 2 ** depth,
            n_categories * self.n_channels, reps=self.depth//2)
        self.final_layer2 = XceptionBlock(
            n_categories * self.n_channels,
            self.n_categories, reps=self.depth - self.depth//2,
            end_with_relu=False)
        self.final_layer = nn.Sequential(
            [self.dropout, self.final_layer1, self.final_layer2])

    def forward(self, image, use_sigmoid=False):
        outputs = []
        inputs = [image]
        assert image.size()[1] == self.n_channels, \
            f'Provided image channels({image.size(1)})'\
            f' do not match with the ones given in the model definition({self.n_channels})'
        assert image.size()[2] % 2**(self.depth - 1) == 0 and\
            image.size()[3] % 2**(self.depth - 1) == 0,\
            f'Image H and W must be divisible by {2**self.depth},'\
            f' but the following size was provided: {image.size()}'


        for block in self.down_sampling_blocks:
            inputs.append(block(inputs[-1]))
        x = None
        outputs = []

        for cnt, (inp, block, upsampler) in enumerate(zip(
                inputs[::-1], self.up_sampling_blocks, self.upsamplers[::-1])):
            x = block(inp, x)
            outputs.append(upsampler(x))
        output = self.final_layer(torch.cat(outputs, dim=1))
        if use_sigmoid:
            output = torch.sigmoid(output)
        return output
