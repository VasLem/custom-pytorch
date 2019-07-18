from torch import nn
import torch

class MultiViewLayer(nn.Module):
    """Input size must be dividable by (field_of_view[0] - 1, field_of_view[1] - 1)
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, field_of_view=3):
        super().__init__()
        orig_out_channels = out_channels
        out_channels = int(out_channels / 3)
        self_out_channels = int(orig_out_channels - 2 * out_channels)
        if not padding:
            padding = int((kernel_size - 1) / 2)
        self.normal_layer = nn.Conv2d(in_channels, self_out_channels, kernel_size, stride=stride,
                                      padding=padding, dilation=dilation, groups=groups, bias=bias)
        if isinstance(field_of_view, int):
            field_of_view = (field_of_view, field_of_view)
        self.maxpooling = nn.MaxPool2d((field_of_view[0] - 1, field_of_view[1] - 1), ceil_mode=False)
                                        # padding=(int((field_of_view[0] - 1 )/ 2), int((field_of_view[1] - 1 )/ 2)))
        self.maxpooled_layer = nn.Conv2d(
            self_out_channels, self_out_channels, kernel_size, stride=stride,
            padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.maxunpooling = nn.ConvTranspose2d(
            self_out_channels, out_channels,
            kernel_size=field_of_view,
            stride=(field_of_view[0] - 1, field_of_view[1] - 1), padding=1, output_padding=1)
        self.maxunpooling_pooled = nn.ConvTranspose2d(
            self_out_channels, out_channels,
            kernel_size=field_of_view,
            stride=(field_of_view[0] - 1, field_of_view[1] - 1), padding=1, output_padding=1)

    def forward(self, batch):
        self_out = self.normal_layer(batch)
        pooled = self.maxpooled_layer(self.maxpooling(self_out))
        unpooled = self.maxunpooling(self_out)
        unpooled_pooled = self.maxunpooling_pooled(pooled)
        pooled_unpooled = self.maxpooling(unpooled)
        return torch.cat((pooled_unpooled, unpooled_pooled, self_out), 1)
