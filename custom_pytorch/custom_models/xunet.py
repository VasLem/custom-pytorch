from torch import nn
import torch
from copy import deepcopy as copy
from abc import abstractmethod
from segmentation_models_pytorch.base.model import Model
from custom_pytorch.custom_layers.custom_xception_with_se import SEXceptionBlock
from segmentation_models_pytorch.common.blocks import Conv2dReLU
import torch.nn.functional as F

class _DecoderBlock(Model):
    def __init__(self, in_channels, out_channels, scale_ratio, *args, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale_ratio = scale_ratio

    @abstractmethod
    def forward(self, inputs):
        pass

class _Downsampler(Model):
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

    @abstractmethod
    def forward(self, inputs):
        pass



class DecodingColumn(Model):
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
        i_ratio = encoder_input_shape[1] / encoder_output_shape[1]
        j_ratio = encoder_input_shape[2] / encoder_output_shape[2]
        if i_ratio != j_ratio:
            raise NotImplementedError('The algorithm expects an encoder that alters uniformly'
                                      ' the spatial characteristics of the input. Different scaling'
                                      ' ratios are not supported')
        self.encoder_ratio = i_ratio
        self.inp_channels = encoder_output_shape[0]
        self.depth = depth
        self.output_shapes = [encoder_output_shape]
        self.column_downsamplers = []
        if self.depth == 0:
            self.column_decoders = [nn.Sequential([])]

        else:
            assert self.depth == 1 or previous_column is not None,\
                'Previous column needs to be provided for depth > 1'
            if previous_column is not None:
                assert len(previous_column.column_decoders) == depth - 1,\
                    'Previous column needs to be 1 depth shorter than this one'
            self.column_decoders = [
                decoder_block_class(encoder_output_shape[0], encoder_input_shape[0],
                                              scale_ratio=self.encoder_ratio)]
            self.output_shapes.append(encoder_input_shape)
            if depth != 1:
                self.column_decoders = self.column_decoders + [
                    decoder_block_class(
                        2 * in_shape[0], out_shape[0],
                        scale_ratio=out_shape[1] / in_shape[1]) for
                    cnt, (in_shape, out_shape) in enumerate(zip(
                        previous_column.output_shapes[:-1], previous_column.output_shapes[1:]))]
                self.column_downsamplers = [
                    downsampler_block_class(2 * in_shape[0], in_shape[0]) for in_shape in
                    previous_column.output_shapes[1:]]
                self.output_shapes.extend(previous_column.output_shapes[1:])
                # self.output_shapes.extend((previous_column.output_shapes[1] * 2, )
                #                           + tuple(previous_column.output_shapes[2:]))
        if self.column_downsamplers:
            self.column_downsamplers = nn.ModuleList(self.column_downsamplers)
        self.column_decoders = nn.ModuleList(self.column_decoders)
        self.initialize()


    def extract_features(self, inputs, previous_column_outputs):
        if self.depth == 0:
            return [inputs]


        features = [inputs]
        for cnt, (decoder, previous_column_output) in enumerate(zip(
                self.column_decoders,  previous_column_outputs)):
                if cnt:
                    previous_column_output = self.column_downsamplers[cnt - 1](
                        previous_column_output)
                decoded = decoder(features[-1])
                features.append(torch.cat((previous_column_output, decoded), dim=1))
        return features

    def forward(self, inputs, previous_column_outputs):
        feats = self.extract_features(inputs, previous_column_outputs)

        return feats[-1]



class XUnet(nn.Module):

    def __init__(self, inp_shape, decoder_block_class: _DecoderBlock,
                 downsampler_block_class: _Downsampler,
                 encoder_blocks_out_shapes, encoder_blocks=None,
                 shared_decoders=False):
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

        :param encoder_blocks_out_shapes: the encoder blocks outputs shapes, as a list of tuples in the
            form (n_channels, height, width)
        :type encoder_blocks_out_shapes: list(tuple(3))
        :param encoder_blocks: the encoder blocks, which are to be given in a way that sequential logic
            can be performed. For example, if the encoder is Inception and the layers 12, 18 and 36 are
            to be used for the encoding, then the blocks will be Inception[:12], Inception[13: 18] and
            Inception[19: 36], for maximum efficiency. It is obvious that each block is expected
            to be callable. If not supplied, a list of the encoded features must be supplied during forwarding.
        :type encoder_blocks: list
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

        decoding_columns = []
        self.shared_decoders = shared_decoders
        for d in range(self.depth):
            if not decoding_columns:
                decoding_columns.append(
                    DecodingColumn(d + 1, decoder_block_class, downsampler_block_class,
                    self.encoder_blocks_in_shapes[d], self.encoder_blocks_out_shapes[d]))
            else:
                decoding_columns.append(
                    DecodingColumn(d + 1, decoder_block_class, downsampler_block_class,
                    self.encoder_blocks_in_shapes[d], self.encoder_blocks_out_shapes[d],
                    decoding_columns[-1]))
        if self.shared_decoders:
            for column in decoding_columns[:-1]:
                for cnt in range(len(column.column_decoders)):
                    if cnt > 0:
                        neg_ind = len(column.column_decoders) - cnt
                        column.column_decoders[cnt] = decoding_columns[-1].column_decoders[-neg_ind]
        final_downsamplers = nn.ModuleList([
            downsampler_block_class(2 * shape[0], shape[0]) for shape
            in self.encoder_blocks_in_shapes])
        decoding_columns = nn.ModuleList(decoding_columns)
        self.decoder = nn.ModuleDict(dict(decoding_columns=decoding_columns,
                                          final_downsamplers=final_downsamplers))

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
        previous_outputs = [input]
        cnt = 0
        for feat, dec in zip(encoded_features[1:], self.decoder['decoding_columns']):
            try:
                previous_outputs = dec.extract_features(feat, previous_outputs)
            except RuntimeError:
                print(f"CUDA Error while handling layer {cnt}")
                raise
            cnt += 1
        features = previous_outputs
        features[1:] = [
            downsampler(feat) for downsampler, feat in
            zip(self.decoder['final_downsamplers'][::-1],features[1:])]
        return features


class SimpleDecoderBlock(_DecoderBlock):
    def __init__(self, inp_channels, out_channels, scale_ratio):
        super().__init__(inp_channels, out_channels, scale_ratio)
        sequence = []
        if scale_ratio > 1:
            sequence.append(nn.UpsamplingBilinear2d(scale_factor=scale_ratio))
        sequence.append(nn.Conv2d(inp_channels, out_channels, 3, padding=1))
        if scale_ratio < 1:
            sequence.append(nn.FractionalMaxPool2d(3, output_ratio=scale_ratio))
        sequence.append(nn.ReLU6(inplace=True))
        self.sequence = nn.Sequential(*sequence)
        self.initialize()

    def forward(self, input):
        return self.sequence(input)

class UnetDecoderBlock(_DecoderBlock):
    def __init__(self, inp_channels, out_channels, scale_ratio, use_batchnorm=True):
        super().__init__(inp_channels, out_channels, scale_ratio)
        self.block = nn.Sequential(
            Conv2dReLU(inp_channels, out_channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm),
            Conv2dReLU(out_channels, out_channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm),
        )
        self.initialize()


    def forward(self, x):
        # x, skip = x
        x = F.interpolate(x, scale_factor=self.scale_ratio, mode='nearest')
        # if skip is not None:
        #     x = torch.cat([x, skip], dim=1)
        x = self.block(x)
        return x

class SEXceptionDecoderBlock(_DecoderBlock):
    def __init__(self, in_channels, out_channels, scale_ratio, *args, **kwargs):
        super().__init__(in_channels, out_channels, scale_ratio)
        self.block = SEXceptionBlock(in_channels, out_channels, 1)
        self.initialize()

    def forward(self, x):
        # x, skip = x
        x = F.interpolate(x, scale_factor=self.scale_ratio, mode='nearest')
        # if skip is not None:
        #     x = torch.cat([x, skip], dim=1)
        x = self.block(x)
        return x


class SimpleDownsamplerBlock(_Downsampler):
    def __init__(self, inp_channels, out_channels):
        super().__init__(inp_channels, out_channels)
        self.sequence = nn.Sequential(
            nn.Conv2d(inp_channels, out_channels, 3, padding=1),
            nn.ReLU6())
        self.initialize()

    def forward(self, input):
        return self.sequence(input)

class SEXceptionDownsamplerBlock(_Downsampler):
    def __init__(self, inp_channels, out_channels):
        super().__init__(inp_channels, out_channels)
        self.sequence = SEXceptionBlock(inp_channels, out_channels, 1)
        self.initialize()

    def forward(self, input):
        return self.sequence(input)

from segmentation_models_pytorch.encoders import get_preprocessing_fn, get_encoder
class SimpleXUnet(XUnet):
    def __init__(self, encoder, sample_input, n_categories, encoder_features_method=None,
                 shared_decoders=False, reversed_features=True, activation='sigmoid'):
        """A simple XUnet

        :param encoder: the encoder to use
        :type encoder: nn.Module or str
        :param encoder_features_method: the method name to use to extract features from the encoder,
         if None the encoder will be just called, defaults to None
        :type encoder_features_method: str
        :param sample_input: the sample input to pass to encoder
        :type sample_input: Tensor
        :param n_categories: the number of categories
        :type n_categories: int
        :param shared_decoders: whether to use shared decoders, defaults to False
        :type shared_decoders: bool, optional
        :param reversed_features: whether to reverse the supplied encoder features, defaults to True
        :type reversed_features: bool, optional
        """
        if isinstance(encoder, str):
            encoder = get_encoder(encoder, 'imagenet')

        inp_shape = sample_input.size()[-3:]
        if encoder_features_method is None:
            encoder_features_method = '__call__'
        feats = getattr(encoder, encoder_features_method)(sample_input)
        if reversed_features:
            feats = feats[::-1]
        out_shapes = [feat.size()[-3:] for feat in feats]
        print(out_shapes)
        super().__init__(inp_shape, SEXceptionDecoderBlock,
                 SEXceptionDownsamplerBlock,
                 out_shapes, shared_decoders=shared_decoders)
        self.encoder_features_method = encoder_features_method
        self.reversed_features = reversed_features
        self.n_categories = n_categories
        self.out_model = nn.Conv2d(inp_shape[0], self.n_categories, 3, padding=1)
        self.encoder = encoder
        if callable(activation) or activation is None:
            self.activation = activation
        elif activation == 'softmax':
            self.activation = nn.Softmax(dim=1)
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            raise ValueError('Activation should be "sigmoid"/"softmax"/callable/None')

    def forward(self, inputs):
        enc_features = getattr(self.encoder, self.encoder_features_method)(inputs)
        if self.reversed_features:
            enc_features = enc_features[::-1]
        ret = super().forward(inputs, enc_features)
        return self.out_model(ret)

    def extract_features(self, input):
        enc_features = getattr(self.encoder, self.encoder_features_method)(input)
        if self.reversed_features:
            enc_features = enc_features[::-1]
        return super().extract_features(input, encoded_features=enc_features)

    def predict(self, x):
        if self.training:
            self.eval()
        with torch.no_grad():
            x = self.forward(x)
            if self.activation:
                x = self.activation(x)
        return x