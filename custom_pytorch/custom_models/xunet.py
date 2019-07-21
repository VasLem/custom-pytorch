from torch import nn
import torch
from copy import deepcopy as copy
from abc import abstractmethod

class _DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scale_ratio, *args, **kwargs):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale_ratio = scale_ratio

    @abstractmethod
    def forward(self, inputs):
        pass

class _Downsampler(nn.Module):
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        self.in_channels = in_channels
        self.out_channels = out_channels

    @abstractmethod
    def forward(self, inputs):
        pass



class DecodingColumn(nn.Module):
    def __init__(self, depth, decoder_block_class: _DecoderBlock,
                 downsampler_block_class: _Downsampler,
                 encoder_input_shape, encoder_output_shape, previous_column=None):
        """

        :param depth: the column depth/height, integer > 0
        :type depth:
        :param decoder_block_class: the Decoder Block class.
            The object needs to be initialized by supplying the following arguments:
            `in_channels`, `out_channels`, `scale_ratio`, where scale ratio refers to
            the spatial ratio that the decoder will expand the provided input and can be float
        :type decoder_block_class: _DecoderBlock
        :param downsampler_block_class: the Downsampler class
            The object needs to be initialized by supplying `in_channels` and `out_channels`
        :type downsampler_block_class: _Downsampler
        :param encoder_input_shape: The input shape to the encoder which produced the input to
            the column, ommitting batch size. The ratio between the input and output shape is
            needed, so the absolute dimensions values will not be considered
        :type encoder_input_shape: tuple(3)
        :param encoder_output_shape: The output shape of the encoder which produced
            the input to the column, ommitting batch size.
            The ratio between the input and output shape is needed,
            so the absolute dimensions values will not be considered
        :type encoder_output_shape: tuple(3)
        :param previous_column: the previous column, which at depth>1 is required and needs
            to have `depth - 1` as depth, defaults to None
        :type previous_column: Column, optional
        """
        super().__init__()
        i_ratio = encoder_output_shape[1] / encoder_input_shape[1]
        j_ratio = encoder_output_shape[2] / encoder_input_shape[2]
        if i_ratio != j_ratio:
            raise NotImplementedError('The algorithm expects an encoder that alters uniformly'
                                      ' the spatial characteristics of the input. Different scaling'
                                      ' ratios are not supported')
        self.encoder_ratio = i_ratio
        self.inp_channels = inp_channels
        self.depth = depth
        self.previous_column = previous_column
        self.output_shapes = [encoder_output_shape]
        if depth == 0:
            self.column = nn.Sequential([])
        else:
            assert depth == 1 or previous_column is not None,\
                'Previous column needs to be provided for depth > 1'
            assert len(previous_column.column) == depth - 1,\
                'Previous column needs to be 1 depth shorter than this one'
            self.column = decoder_block_class(encoder_output_shape[0], encoder_input_shape[0],
                                              scale_ratio=self.encoder_ratio)
            self.output_shapes.append(encoder_input_shape)
            if depth != 1:
                self.column_decoders = [self.column] + [
                    decoder_block_class(in_shape[0], out_shape[0], scale_ratio=out_shape[1] / in_shape[1]) for
                    in_shape, out_shape in zip(
                        previous_column.output_shapes[:-1], previous_column.output_shapes[1:])]
                self.column_decoders = nn.ModuleList(self.column)
                self.column_downsamplers = nn.ModuleList([
                    downsampler_block_class(2 * in_shape[0], in_shape[0]) for in_shape in
                    previous_column.output_shapes])

    def extract_features(self, inputs, previous_column_outputs):
        if self.depth == 0:
            return [inputs]
        features = [inputs, self.column_decoders[0][inputs]]
        if self.previous_column:
            for decoder, downsampler, previous_column_output in zip(
                self.column_decoders[1:], self.column_downsamplers, previous_column_outputs):
                features.append(decoder(
                    torch.cat((downsampler(previous_column_output),
                    features[-1]))))
        return features

    def forward(self, inputs, previous_column_outputs):
        return self.extract_features(inputs, previous_column_outputs)[-1]



class ColumnsDiagonalJoiner:
    def __init__(self, n_channels, columns, depth):
        self.depth = depth
        if depth == 0:
            self.out_layers = n_channels + sum(
                [column.column[cnt].out_channels for cnt, column
                 in enumerate(columns[self.depth:])])
        else:
            self.out_layers = columns[self.depth - 1].inp_channels + sum(
                [column.column[cnt].out_channels for cnt, column
                 in enumerate(columns[self.depth:])])

    def __call__(self, inputs, columns_outputs):
        total = [[inputs]] + columns_outputs
        return torch.cat([el[cnt] for cnt, el in enumerate(total[self.depth:])], dim=1)


class XUnet(nn.Module):
    """An expanded UNet representation in the following form:

    I -e1-> R1 -e2------> R2
    |        | -----      |
    |       d1      |     d2
    |        |      |     |
    |------Concat   -D->Concat
             |            |
             |            d1
             |            |
             -----D---->Concat--> Output
    It is like having multiple sub-unets, up to the original, while interconnecting
    them by concatenation. e_i are the encoder blocks, d_i are the decoder blocks,
    D is the downsampler black, which halfs down the provided input channels, and
    R_i are the encoded outputs. This is basically a decoding architecture, while the
    encoder can be any of the already known and state of the art networks.
    """
    def __init__(self, inp_shape, n_categories, decoder_block_class: _DecoderBlock,
                 downsampler_block_class: _Downsampler,
                 encoder_blocks, encoder_blocks_out_shapes):
        super().__init__()
        self.n_channels = inp_shape[0]
        self.n_categories = n_categories
        self.depth = len(encoder_blocks)
        self.encoder_blocks = encoder_blocks
        self.encoder_blocks_out_shapes = encoder_blocks_out_shapes
        self.encoder_blocks_in_shapes = [inp_shape] + encoder_blocks_out_shapes[:-1]
        self.decoding_columns = []
        for d in range(self.depth):
            if not self.decoding_columns:
                self.decoding_columns.append(
                    DecodingColumn(d + 1, decoder_block_class, downsampler_block_class,
                    self.encoder_blocks_in_shapes[d], self.encoder_blocks_out_shapes[d]))
            else:
                self.decoding_columns.append(
                   DecodingColumn(d + 1, decoder_block_class, downsampler_block_class,
                    self.encoder_blocks_in_shapes[d], self.encoder_blocks_out_shapes[d],
                    self.decoding_columns[-1]))
        self.decoding_columns = nn.ModuleList(self.decoding_columns)

    def forward(self, input):
        x = input
        previous_outputs = None
        for enc, dec in zip(self.encoder_blocks, self.decoding_columns):
            x = enc(x)
            previous_outputs = dec.extract_features(x, previous_outputs)
        return previous_outputs[-1]
