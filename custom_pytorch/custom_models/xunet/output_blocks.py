from custom_pytorch.custom_layers.custom_xception_with_se import SEXceptionBlock
import torch.nn.functional as F

class SEXceptionOutputBlock(_OutputBlock):
    def __init__(self, in_channels, out_channels):
        super().__init__(in_channels, out_channels)
        self.block = SEXceptionBlock(in_channels, out_channels, 1)
        self.initialize()

    def forward(self, x):
        # x, skip = x
        x = F.interpolate(x, scale_factor=self.scale_ratio, mode='nearest')
        # if skip is not None:
        #     x = torch.cat([x, skip], dim=1)
        x = self.block(x)
        return x
