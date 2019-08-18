import numpy as np
from torch import nn
import torch
from custom_pytorch.custom_layers.base import Model
from custom_pytorch.custom_layers.custom_xception_block import XceptionBlock
from random import shuffle


class ExpandedXceptionBlock(nn.Module):
    def __init__(self, inp_channels, out_channels, depth, stride=1):
        super().__init__()
        self.inp_channels = inp_channels
        self.out_channels = out_channels

        if out_channels > inp_channels:
            filters_hierarchy = np.linspace(
                inp_channels, out_channels, depth + 2).astype(int)
        else:
            filters_hierarchy = np.linspace(
                out_channels, inp_channels, depth + 2).astype(int)[::-1]

        inp_channels_group = filters_hierarchy[:-1].tolist()
        out_channels_group = filters_hierarchy[1:].tolist()
        strides_num = np.log2(stride)
        assert strides_num == int(
            strides_num), "Only power of 2 strides are accepted"
        strides_num = int(strides_num)
        strides = [2] * strides_num + [1] * \
            (len(inp_channels_group) - strides_num)
        shuffle(strides)
        # print(inp_channels, out_channels, reps, filters_hierarchy)
        rep = []
        for in_filt, out_filt, stride in zip(inp_channels_group, out_channels_group, strides):
            rep.append(nn.Sequential(
                XceptionBlock(in_filt, in_filt, 1),
                XceptionBlock(in_filt,
                              out_filt,
                              strides=stride, reps=2),
                XceptionBlock(out_filt, out_filt, 1)))
        self.sequence = nn.Sequential(*rep)

    def forward(self, input):
        return self.sequence(input)


class EncodingBlock(nn.Module):
    def __init__(self, n_channels, depth, resolution=1):
        super().__init__()
        self.inp_channels = int(n_channels * resolution * 2 ** depth)
        self.out_channels = int(n_channels * resolution * 2 ** (depth + 1))
        self.xception_block = ExpandedXceptionBlock(self.inp_channels, self.out_channels, depth,
                                                    stride=2)

    def forward(self, image):
        x = self.xception_block(image)
        return x


class DecodingBlock(nn.Module):
    def __init__(self, n_channels, depth, inv_depth, resolution=1, use_transpose=False):
        super().__init__()
        if depth != 0:
            self.inp_channels = int(
                n_channels * 2 ** (inv_depth) * 2 * resolution)
        else:
            self.inp_channels = int(n_channels * 2 ** (inv_depth) * resolution)
        self.out_channels = int(
            n_channels * 2 ** max(0, (inv_depth - 1)) * resolution)
        if use_transpose:
            self.upsampler = nn.ConvTranspose2d(
                self.inp_channels // 2, self.inp_channels // 2, (3, 3), 2, 1, 1)
        else:
            self.upsampler = nn.UpsamplingNearest2d(scale_factor=2)
        self.xception_block = ExpandedXceptionBlock(
            self.inp_channels, self.out_channels, inv_depth)

    def forward(self, image, previous_output=None):
        if previous_output is not None:
            previous_output = self.upsampler(previous_output)
            image = torch.cat((image, previous_output), dim=1)
        return self.xception_block(image)


class SamplingSegmentationV3(Model):
    def __init__(self, n_channels, n_categories, depth, resolution=1):
        super().__init__()
        self.n_categories = n_categories
        self.n_channels = n_channels
        self.depth = depth
        self.init_block = XceptionBlock(n_channels, int(
            n_channels * resolution), int(resolution))
        self.init_channels = int(n_channels * resolution)
        self.decoding_blocks = nn.ModuleList([DecodingBlock(n_channels, d,
                                                            self.depth - d, resolution)
                                              for d in range(self.depth + 1)])
        self.encoding_blocks = nn.ModuleList([EncodingBlock(n_channels, d, resolution)
                                              for d in range(self.depth)])
        self.upsamplers = nn.ModuleList([
            nn.Sequential(
                nn.UpsamplingNearest2d(scale_factor=2 ** d),
                ExpandedXceptionBlock(self.decoding_blocks[::-1][d].out_channels,
                                      self.decoding_blocks[::-1][d].out_channels, d)
            )

            for d in range(self.depth + 1)])
        # self.downsamplers = nn.ModuleList([XceptionBlock()])
        self.dropout = nn.Dropout2d(p=0.2, inplace=False)
        self.final_layer1_inp_channels = sum(
            [block.out_channels for block in self.decoding_blocks])
        final_layers = [
            ExpandedXceptionBlock(self.final_layer1_inp_channels, n_categories * resolution,
                                  depth=self.depth),
            nn.Conv2d(n_categories * resolution, n_categories, 1)]
        self.final_layer = nn.Sequential(*final_layers)
        self.initialize()

    def forward(self, image):
        outputs = []
        encoded_inputs = [self.init_block(image)]
        assert image.size()[1] == self.n_channels, \
            f'Provided image channels({image.size(1)})'\
            f' do not match with the ones given in the model definition({self.n_channels})'
        assert image.size()[2] % 2**(self.depth - 1) == 0 and\
            image.size()[3] % 2**(self.depth - 1) == 0,\
            f'Image H and W must be divisible by {2**self.depth},'\
            f' but the following size was provided: {image.size()}'

        for encoding_block in self.encoding_blocks:
            encoded_inputs.append(encoding_block(encoded_inputs[-1]))
        x = None
        outputs = []
        for cnt, (encoded_inp, decoding_block, upsampler) in enumerate(zip(
                encoded_inputs[::-1], self.decoding_blocks, self.upsamplers[::-1])):
            x = decoding_block(encoded_inp, x)
            outputs.append(upsampler(x))
        x = torch.cat(outputs, dim=1)
        x = self.dropout(x)
        x = self.final_layer(x)
        return x

    def predict(self, image):
        output = self(image)
        output = nn.Sigmoid()(output)
        return output


if __name__ == '__main__':
    depth = 5
    resolution = 1
    net = SamplingSegmentationV3(3, 4, depth, resolution)
    from numpy.random import random
    net.eval()
    net(torch.from_numpy(random((1, 3, 128, 128))).float())
    from custom_pytorch.custom_utils import submodules_number, params_number
    print("Depth:", depth, ', Resolution:', resolution)
    print('Submodules number:', submodules_number(net))
    print("Parameters number:", params_number(net))
