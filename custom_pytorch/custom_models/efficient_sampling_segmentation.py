import numpy as np
from torch import nn
import torch
import torch.nn.functional as F
from custom_pytorch.custom_layers.custom_xception_block import XceptionBlock, SeparableConv2d


class DownSamplingBlock(nn.Module):
    def __init__(self, inp_channels):
        super().__init__()
        self.inp_channels = inp_channels
        self.out_channels = inp_channels * 4
        self.xception_block = XceptionBlock(self.inp_channels,
                                            self.out_channels,
                                            strides=2, reps=3)

    def forward(self, image):
        return self.xception_block(image)


class UpSamplingBlock(nn.Module):
    def __init__(self, inp_channels, reduce=True, scale_factor=2, reps=3, use_transpose=True):
        super().__init__()
        # self.upsampler = nn.UpsamplingNearest2d(scale_factor=scale_factor)
        self.inp_channels = inp_channels
        if reduce:
            self.out_channels = inp_channels // scale_factor ** 2
        else:
            self.out_channels = inp_channels
        if scale_factor > 1:
            if use_transpose:
                if scale_factor % 2 == 1:
                    raise NotImplementedError
                opts_stride = dict(
                    stride=scale_factor, kernel_size=scale_factor + 1, padding=1, output_padding=1)
                self.upsampler = nn.ConvTranspose2d(
                    inp_channels, self.out_channels, **opts_stride)
            else:
                self.upsampler = nn.Sequential([nn.UpsamplingNearest2d(scale_factor=scale_factor),
                    XceptionBlock(self.inp_channels, self.out_channels,
                                                reps=reps)])
        elif scale_factor == 1:
            self.upsampler = XceptionBlock(self.inp_channels, self.out_channels,
                                                reps=reps)


        assert self.out_channels > 0


    def forward(self, inputs):
        if isinstance(inputs, tuple) or isinstance(inputs, list):
            inputs = torch.cat(inputs, dim=1)
        return self.upsampler(inputs)


class UpSamplingColumn(nn.Module):
    def __init__(self, inp_layers, depth):
        super().__init__()
        assert inp_layers // 4 ** depth >= 1
        self.inp_layers = inp_layers
        self.depth = depth
        self.column = nn.ModuleList([UpSamplingBlock(inp_layers // 4 ** d) for d in range(depth)])
        self.layer = nn.Sequential(self.column)

    def forward(self, inputs):
        self.outputs = [inputs]

        for el in self.column:
            self.outputs.append(el(self.outputs[-1]))
        return self.outputs[-1]


class ColumnsDiagonalJoiner:
    def __init__(self, n_channels, columns, depth):
        self.depth = depth
        if depth == 0:
            self.out_layers = n_channels + sum(
                [column.column[cnt].out_channels for cnt, column
                 in enumerate(columns[self.depth:])])
        else:
            self.out_layers = columns[self.depth - 1].inp_layers+ sum(
                    [column.column[cnt].out_channels for cnt, column
                    in enumerate(columns[self.depth:])])


    def __call__(self, inputs, columns_outputs):
        total = [[inputs]] + columns_outputs
        return torch.cat([el[cnt] for cnt, el in enumerate(total[self.depth:])], dim=1)





class SamplingSegmentationV4(nn.Module):
    def __init__(self, n_channels, n_categories, depth):
        super().__init__()
        self.n_categories = n_categories
        self.n_channels = n_channels
        self.depth = depth
        self.up_sampling_columns = nn.ModuleList([
            UpSamplingColumn(n_channels * 4 ** d, d) for d in range(1, self.depth + 1)])
        self.diagonal_joiner = [ColumnsDiagonalJoiner(
            n_channels, self.up_sampling_columns, d) for d in range(self.depth)]

        self.down_sampling_blocks = nn.ModuleList([DownSamplingBlock(n_channels * 4 ** d)
                                                   for d in range(self.depth)])
        self.upsamplers = nn.ModuleList([UpSamplingBlock(joiner.out_layers, False,
            scale_factor=2 ** cnt) for cnt, joiner in enumerate(self.diagonal_joiners)])
        self.dropout = nn.Dropout2d()
        self.features_layers = sum(
            [block.out_layers for block in self.diagonal_joiners])
        self.final_layer = XceptionBlock(
            self.features_layers,
            int(n_categories), reps=2)
        self.sampled_features = []

    def forward(self, image):
        assert image.size()[1] == self.n_channels, \
            f'Provided image channels({image.size(1)})'\
            f' do not match with the ones given in the model definition({self.n_channels})'
        assert image.size()[2] % 4**(self.depth) == 0 and\
            image.size()[3] % 4**(self.depth) == 0,\
            f'Image H and W must be divisible by {4**self.depth},'\
            f' but the following size was provided: {image.size()}'
        x = image
        downsampled = []
        for block in self.down_sampling_blocks:
            downsampled.append(block(x))
            x = downsampled[-1]
        for column, sample in zip(self.up_sampling_columns, downsampled):
            column(sample)
        columns_outputs = [column.outputs for column in self.up_sampling_columns]
        joined = []
        for joiner in self.diagonal_joiners:
            joined.append(joiner(image, columns_outputs))
        self.sampled_features = []
        for stack, upsampler in zip(joined, self.upsamplers):
            self.sampled_features.append(upsampler(stack))
        out = self.final_layer(self.dropout(torch.cat(self.sampled_features, dim=1)))
        return out

if __name__ == '__main__':
    depth = 3
    net = SamplingSegmentationV4(3, 4, depth)
    from numpy.random import random
    import time
    import os
    dummy_inp = torch.from_numpy(random((1, 3, 4 ** depth, 4 ** depth))).float()
    t0 = time.time()
    net(dummy_inp)
    print('Time for execution:', time.time() - t0)
    model_parameters = filter(lambda p: p.requires_grad, net.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f'#Parameters for depth {depth}:', params)
    import hiddenlayer as hl
    graph = hl.build_graph(net, dummy_inp)
    graph = graph.build_dot()
    try:
        os.makedirs('graphs')
    except:
        pass
    graph.render('graphs/SamplingSegmentationV4', view=True, format='svg')