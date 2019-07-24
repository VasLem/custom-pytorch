from .custom_xception_block import XceptionBlock
from custom_pytorch.layers import SqueezeAndExcitation

class SEXceptionBlock(XceptionBlock):
    """Exception Block, with the Squeeze and Excitation variant. Folowing SE-PRE strategy.
    The SE variant is applied only when the skip connection is activated, which happens if the
    numbers of input layers and output layers differ, or when the stride is != 1.
    """
    def __init__(self, in_filters, out_filters, reps=1, strides=1,
                 start_with_relu=False, end_with_relu=True):
        super().__init__(in_filters=in_filters, out_filters=out_filters,
                         reps=reps, start_with_relu=start_with_relu, end_with_relu=end_with_relu)
        if out_filters != in_filters or strides != 1:
            self.se_block = SqueezeAndExcitation(in_filters, reduction=min(in_filters, 16))

    def forward(self, inp):
        """Following SE-PRE block connections
        """
        x = inp
        if self.skip is not None:
            x = self.se_block(inp)
        x = self.rep(x)
        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp
        # print(self.in_filters, self.out_filters, x.size(), skip.size())
        x += skip
        return x

