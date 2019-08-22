import numpy as np
from torch import nn
from custom_pytorch.custom_layers.separable_conv2relu import SeparableConv2dReLU


class XceptionBlock(nn.Module):
    def __init__(self, in_filters, out_filters, reps=1, strides=1,
                 start_with_relu=False, end_with_relu=True, apply_smooth_transform=False,
                 expand_first=True):
        super().__init__()

        reps = max(1, reps)
        self.ending_relu = None
        if end_with_relu:
            self.ending_relu = nn.LeakyReLU(inplace=False)
        self.skip = None
        if in_filters != out_filters or strides != 1:
            self.skip = SeparableConv2dReLU(in_filters, out_filters,
                                            3, stride=1, padding=1)

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
                rep.append(SeparableConv2dReLU(in_filt, out_filt,
                                               3, stride=1, padding=1))

        else:
            if expand_first:
                rep.append(SeparableConv2dReLU(in_filters, out_filters,
                                               3, stride=1, padding=1))

                for _ in range(reps - 1):
                    rep.append(SeparableConv2dReLU(out_filters, out_filters))

            else:
                for _ in range(reps - 1):
                    rep.append(SeparableConv2dReLU(in_filters, in_filters))

                rep.append(SeparableConv2dReLU(in_filters, out_filters,
                                               3, stride=1, padding=1))

        if start_with_relu:
            rep = [nn.LeakyReLU(inplace=False)] + rep

        if strides != 1:
            rep.append(nn.MaxPool2d(3, strides, 1))

        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            x = x + skip
        else:
            x = x + inp
        if self.ending_relu is not None:
            x = self.ending_relu(x)
        return x
