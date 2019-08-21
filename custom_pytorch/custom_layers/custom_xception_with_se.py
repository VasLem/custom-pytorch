from .custom_xception_block import XceptionBlock
from custom_pytorch.layers import SqueezeAndExcitation


class SEXceptionBlock(XceptionBlock):
    """Exception Block, with the Squeeze and Excitation variant. Folowing SE-PRE strategy:
    The SE variant is applied on the non-skipping branch, before applying the exception block.
    """

    def __init__(self, in_filters, out_filters, reps=1, strides=1,
                 start_with_relu=False, end_with_relu=True, apply_smooth_transform=False,
                 expand_first=True, reduction=4):
        super().__init__(in_filters=in_filters, out_filters=out_filters,
                         reps=reps, start_with_relu=start_with_relu,
                         end_with_relu=end_with_relu, apply_smooth_transform=apply_smooth_transform,
                         expand_first=expand_first, strides=strides)
        self.se_block = SqueezeAndExcitation(
            in_filters, reduction=min(in_filters, reduction))

    def forward(self, inp):
        """Following SE-PRE block connections
        """
        x = inp
        x = self.se_block(inp)
        x = self.rep(x)
        if self.skip is not None:
            skip = self.skip(inp)
            x = x + skip
        else:
            x = x + inp
        if self.ending_relu is not None:
            x = self.ending_relu(x)
        return x
