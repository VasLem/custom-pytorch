
from .base import XUnet
from .decoder_blocks import SEXceptionDecoderBlock
from .downsampler_blocks import SEXceptionDownsamplerBlock
from .output_blocks import SEXceptionOutputBlock


class SEXceptionXUnet(XUnet):
    def __init__(self, encoder, sample_input, n_categories, encoder_features_method=None,
                 shared_decoders=False, reversed_features=True, activation='sigmoid'):
        """A XUnet with decoders, outputs and downsamplers SEXception blocks

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

        super().__init__(encoder=encoder,  decoder_block_class=SEXceptionDecoderBlock,
                         downsampler_block_class=SEXceptionDownsamplerBlock,
                         output_block_class=SEXceptionOutputBlock, sample_input=sample_input,
                         n_categories=n_categories, encoder_features_method=encoder_features_method,
                         shared_decoders=shared_decoders, reversed_features=reversed_features,
                         activation=activation)
