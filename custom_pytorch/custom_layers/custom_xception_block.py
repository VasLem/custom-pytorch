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
    def __init__(self, in_filters, out_filters, reps=1, strides=1,
                 start_with_relu=False, end_with_relu=True, apply_smooth_transform=False,
                 expand_first=True):
        super().__init__()
        reps = max(1, reps)
        if out_filters != in_filters or strides != 1:
            skip = [nn.Conv2d(in_filters, out_filters,
                                  1, stride=strides, bias=False)]
            if end_with_relu:
                skip.append(nn.BatchNorm2d(out_filters))
            self.skip = nn.Sequential(*skip)
        else:
            self.skip = None

        rep = []
        self.in_filters = in_filters
        self.out_filters = out_filters
        if apply_smooth_transform:
            if out_filters > in_filters:
                filters_hierarchy = np.linspace(
                    in_filters, out_filters, reps + 1).astype(int)
            else:
                filters_hierarchy = np.linspace(
                    out_filters, in_filters, reps + 1).astype(int)[::-1]


            in_filters_group = filters_hierarchy[:-1].tolist()
            out_filters_group = filters_hierarchy[1:].tolist()
            # print(in_filters, out_filters, reps, filters_hierarchy)

            for in_filt, out_filt in zip(in_filters_group, out_filters_group):
                rep.append(SeparableConv2d(in_filt, out_filt,
                                        3, stride=1, padding=1, bias=False))
                rep.append(nn.LeakyReLU(inplace=True))
                rep.append(nn.BatchNorm2d(out_filt))
        else:
            if expand_first:
                rep.append(SeparableConv2d(in_filters, out_filters,
                                        3, stride=1, padding=1, bias=False))
                rep.append(nn.LeakyReLU(inplace=True))
                rep.append(nn.BatchNorm2d(out_filters))
                for _ in range(reps - 1):
                    rep.append(SeparableConv2d(out_filters, out_filters))
                    rep.append(nn.LeakyReLU(inplace=True))
                    rep.append(nn.BatchNorm2d(out_filters))
            else:
                for _ in range(reps - 1):
                    rep.append(SeparableConv2d(in_filters, in_filters))
                    rep.append(nn.LeakyReLU(inplace=True))
                    rep.append(nn.BatchNorm2d(in_filters))
                rep.append(SeparableConv2d(in_filters, out_filters,
                                        3, stride=1, padding=1, bias=False))
                rep.append(nn.LeakyReLU(inplace=True))
                rep.append(nn.BatchNorm2d(out_filters))

        if start_with_relu:
            rep = rep[:-2]
            rep = [nn.LeakyReLU(inplace=True), nn.BatchNorm2d(in_filters)] + rep

        elif not end_with_relu:
            rep = rep[:-2]

        if strides != 1:
            rep.append(nn.MaxPool2d(3, strides, 1))
        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            x += skip
        return x

