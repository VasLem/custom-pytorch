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
    def __init__(self, in_filters, out_filters, reps, strides=1, start_with_relu=True, grow_first=True):
        super().__init__()

        if out_filters != in_filters or strides != 1:
            self.skip = nn.Conv2d(in_filters, out_filters,
                                  1, stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip = None

        self.relu = nn.ReLU(inplace=False)
        rep = []

        filters = in_filters
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters,
                                       3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters

        for _ in range(reps-1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters, filters,
                                       3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(filters))

        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters,
                                       3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

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

        x += skip
        return x


class SamplingBlock(nn.Module):
    def __init__(self, inp_channels, out_channels, n_categories, depth):
        super().__init__()
        self.n_categories = n_categories
        self.inp_channels = inp_channels
        self.out_channels = out_channels
        self.xception_block1 = XceptionBlock(
            n_categories * inp_channels, n_categories * out_channels, reps=depth,
            start_with_relu=True,
            grow_first=True)
        self.upsampler = nn.UpsamplingNearest2d(scale_factor=2)
        self.xception_block2 = XceptionBlock(
            2 * n_categories * inp_channels, n_categories * out_channels, reps=depth,
            start_with_relu=True,
            grow_first=True)

    def forward(self, image, previous_output=None, with_sigmoid=False):
        assert image.size()[1] == self.inp_channels, \
            f'Provided image channels({image.size(1)}) do not '\
            f'match with the ones given in the model definition({self.inp_channels})'
        image = image.repeat((1, self.n_categories, 1, 1))
        if previous_output is None:
            x = self.xception_block1(image)

        else:
            previous_output = self.upsampler(previous_output)
            x = self.xception_block2(
                torch.cat((image, previous_output), dim=1))
        if with_sigmoid:
            return torch.sigmoid(x)
        return x


class SamplingSegmentation(nn.Module):
    def __init__(self, n_channels, n_categories, depth):
        super().__init__()
        self.n_categories = n_categories
        self.n_channels = n_channels
        self.depth = depth + 1
        self.blocks = nn.ModuleList([SamplingBlock(n_channels,
                                                   n_channels,
                                                   n_categories, 5)
                                     for cnt, d in enumerate(reversed(range(self.depth)))])
        self.final_layer1 = XceptionBlock(
            n_categories * self.n_channels * self.depth + self.n_channels,
            n_categories * self.n_channels, reps=self.depth//2,
            start_with_relu=False, grow_first=True)
        self.final_layer2 = XceptionBlock(
            n_categories * self.n_channels,
            self.n_categories, reps=self.depth - self.depth//2,
            start_with_relu=True, grow_first=False)
        self.upsamplers = nn.ModuleList([nn.UpsamplingNearest2d(
            scale_factor=2**d) for d in reversed(range(self.depth))])

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
        for block in self.blocks[:-1]:
            inputs.append(F.interpolate(
                inputs[-1], scale_factor=0.5))
        x = None
        for (inp, block, upsampler) in zip(
                inputs[::-1], self.blocks, self.upsamplers):
            x = block(inp, x)
            outputs.append(upsampler(x))
        output = self.final_layer2(self.final_layer1(torch.cat([inputs[0]] + outputs, dim=1)))
        if use_sigmoid:
            output = torch.sigmoid(output)
        return output
