import numpy as np
from torch import nn
import torch
import torch.nn.functional as F
from custom_pytorch.custom_layers.base import Model
from custom_pytorch.custom_layers.custom_xception_with_se import SEXceptionBlock
from custom_pytorch.custom_layers.custom_xception_block import XceptionBlock

class ExpandedSEXceptionBlock(nn.Module):
    def __init__(self, inp_channels, out_channels, depth, stride=1):
        super().__init__()
        self.inp_channels = inp_channels
        self.out_channels = out_channels
        self.sequence = nn.Sequential(
            SEXceptionBlock(self.inp_channels, self.inp_channels, 2),
            SEXceptionBlock(self.inp_channels,
                                            self.out_channels,
                                            strides=stride, reps=depth),
            SEXceptionBlock(self.out_channels, self.out_channels, 2))

    def forward(self, input):
        return self.sequence(input)


class EncodingBlock(nn.Module):
    def __init__(self, n_channels, depth, resolution=1):
        super().__init__()
        self.inp_channels = int(n_channels * resolution * 2 ** depth)
        self.out_channels = int(n_channels * resolution * 2 ** (depth + 1))
        self.xception_block = ExpandedSEXceptionBlock(self.inp_channels, self.out_channels, depth,
                                                      stride=2)


    def forward(self, image):
        return self.xception_block(image)


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
        self.xception_block = ExpandedSEXceptionBlock(self.inp_channels, self.out_channels, inv_depth)

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
        self.init_block = XceptionBlock(n_channels, int(n_channels * resolution), int(resolution))
        self.init_channels = int(n_channels * resolution)
        self.decoding_blocks = nn.ModuleList([DecodingBlock(n_channels, d,
                                                                 self.depth - d, resolution)
                                                 for d in range(self.depth + 1)])
        self.encoding_blocks = nn.ModuleList([EncodingBlock(n_channels, d, resolution)
                                                   for d in range(self.depth)])
        self.upsamplers = nn.ModuleList([
            nn.Sequential(
                # (nn.ConvTranspose2d(self.decoding_blocks[::-1][d].out_channels,
                #             self.decoding_blocks[::-1][d].out_channels, 2 ** d + 1,
                #             stride=2 ** d,
                #             padding=1, output_padding=1)
                # if d > 0 else nn.Conv2d(self.decoding_blocks[::-1][d].out_channels,
                #             self.decoding_blocks[::-1][d].out_channels, 1)),
                nn.UpsamplingNearest2d(scale_factor=2 ** d),
                ExpandedSEXceptionBlock(self.decoding_blocks[::-1][d].out_channels,
                                        self.decoding_blocks[::-1][d].out_channels, d)
                         )

            for d in range(self.depth + 1)])
        # self.downsamplers = nn.ModuleList([SEXceptionBlock()])
        self.dropout = nn.Dropout2d(p=0.2)
        self.final_layer1_inp_channels = sum(
            [block.out_channels for block in self.decoding_blocks])
        self.final_layer1 = SEXceptionBlock(
            self.final_layer1_inp_channels,
            int(n_categories * resolution * self.n_channels), reps=int(resolution * self.depth // 2))
        self.final_layer2 = XceptionBlock(
            int(n_categories * resolution * self.n_channels),
            self.n_categories, reps=int(resolution * self.depth // 2),
            end_with_relu=False)
        self.initialize()

    def forward(self, image):
        outputs = []
        inputs = [self.init_block(image)]
        assert image.size()[1] == self.n_channels, \
            f'Provided image channels({image.size(1)})'\
            f' do not match with the ones given in the model definition({self.n_channels})'
        assert image.size()[2] % 2**(self.depth - 1) == 0 and\
            image.size()[3] % 2**(self.depth - 1) == 0,\
            f'Image H and W must be divisible by {2**self.depth},'\
            f' but the following size was provided: {image.size()}'

        for block in self.encoding_blocks:
            inputs.append(block(inputs[-1]))
        x = None
        outputs = []
        for cnt, (inp, block, upsampler) in enumerate(zip(
                inputs[::-1], self.decoding_blocks, self.upsamplers[::-1])):
            x = block(inp, x)
            outputs.append(upsampler(x))
        output = self.final_layer2(self.final_layer1(
            self.dropout(torch.cat(outputs, dim=1))))
        return output

    def predict(self, image):
        output = self(image)
        output = nn.Sigmoid()(output)
        return output


if __name__ == '__main__':
    net = SamplingSegmentationV3(3, 4, 6, 2)
    from numpy.random import random
    net.eval()
    net(torch.from_numpy(random((1, 3, 128, 128))).float())