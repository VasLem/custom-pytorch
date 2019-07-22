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
        self.column_downsamplers = []
        if depth == 0:
            self.column_decoders = [nn.Sequential([])]

        else:
            assert depth == 1 or previous_column is not None,\
                'Previous column needs to be provided for depth > 1'
            assert len(previous_column.column) == depth - 1,\
                'Previous column needs to be 1 depth shorter than this one'
            self.column_decoders = [
                decoder_block_class(encoder_output_shape[0], encoder_input_shape[0],
                                              scale_ratio=self.encoder_ratio)]
            self.output_shapes.append(encoder_input_shape)
            if depth != 1:
                self.column_decoders = self.column_decoders + [
                    decoder_block_class(in_shape[0], out_shape[0], scale_ratio=out_shape[1] / in_shape[1]) for
                    in_shape, out_shape in zip(
                        previous_column.output_shapes[:-1], previous_column.output_shapes[1:])]
                self.column_downsamplers = [
                    downsampler_block_class(2 * in_shape[0], in_shape[0]) for in_shape in
                    previous_column.output_shapes]
        if self.column_downsamplers:
            self.column_downsamplers = nn.ModuleList(self.column_downsamplers)
        self.column_decoders = nn.ModuleList(self.column)


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



class XUnet(nn.Module):

    def __init__(self, inp_shape, decoder_block_class: _DecoderBlock,
                 downsampler_block_class: _Downsampler,
                 encoder_blocks_out_shapes, encoder_blocks=None, shared_decoders=False):
        """
        An expanded/extreme UNet representation in the following form:

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

        :param inp_shape: the input shape to compare the rest, a tuple (n_channels, height, width).
            The height and width will only be considered for creating scaling ratios
        :type inp_shape: tuple(3, 3)
        :param decoder_block_class: the decoder block class
        :type decoder_block_class: _DecoderBlock
        :param downsampler_block_class: the downsampler block class
        :type downsampler_block_class: _Downsampler
        :param encoder_blocks: the encoder blocks, which are to be given in a way that sequential logic
            can be performed. For example, if the encoder is Inception and the layers 12, 18 and 36 are
            to be used for the encoding, then the blocks will be Inception[:12], Inception[13: 18] and
            Inception[19: 36], for maximum efficiency. It is obvious that each block is expected
            to be callable. If not supplied, a list of the encoded features must be supplied during forwarding.
        :type encoder_blocks: list
        :param encoder_blocks_out_shapes: the encoder blocks outputs shapes, as a list of tuples in the
            form (n_channels, height, width)
        :type encoder_blocks_out_shapes: list(tuple(3))
        :param shared_decoders: whether the decoders in each decoder column are to be shared, something
            that will greatly reduce the proposed network, but will most probably increase exponentially
            training time and may worsen the network efficiency. Defaults to False.
        :type shared_decoders: bool
        """
        super().__init__()
        self.n_channels = inp_shape[0]
        self.depth = len(encoder_blocks_out_shapes)
        self.encoder_blocks = encoder_blocks
        self.encoder_blocks_out_shapes = encoder_blocks_out_shapes
        self.encoder_blocks_in_shapes = [inp_shape] + encoder_blocks_out_shapes[:-1]
        self.decoding_columns = []
        self.shared_decoders = shared_decoders
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
        if self.shared_decoders:
            for column in self.decoding_columns[:-1]:
                for cnt in range(len(column.column_decoders)):
                    column.column_decoders[cnt] = self.decoding_columns[-1].column_decoders[cnt]
        self.decoding_columns = nn.ModuleList(self.decoding_columns)

    def forward(self, input, encoded_features=None):
        """Returns the last output of the last decoding column,
        which has the same shape as the input
        """
        return self.extract_features(input)[-1]

    def extract_features(self, input, encoded_features=None):
        """This will return the last decoding column outputs,
        it will produce more tightly convolved features than the ones provided, interesting
        results may arise if they are compared, it is currently assumed that an invariance in
        scale may be a positive outcome.
        :return: Features of same shape as the ones originally provided
        """
        if self.encoder_blocks is None:
            assert encoded_features is not None, 'The encoded features list must be supplied'
            encoded_features = [input] + encoded_features
        else:
            encoded_features = [input]
            for block in self.encoder_blocks:
                encoded_features.append(block(encoded_features[-1]))
        previous_outputs = None
        for feat, dec in zip(encoded_features[1:], self.decoding_columns):
            previous_outputs = dec.extract_features(feat, previous_outputs)
        return ([encoded_features[-1]] + previous_outputs)[::-1]

