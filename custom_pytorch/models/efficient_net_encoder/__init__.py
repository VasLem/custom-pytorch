from torch import nn
import torch
from efficientunet import EfficientNet
from efficientunet.efficientunet import get_blocks_to_be_concat
from efficientunet.efficientnet import _get_model_by_name

class Swish(nn.Module):
    def __init__(self, name=None):
        super().__init__()
        self.name = name

    def forward(self, x):
        return x * torch.sigmoid(x)

class Encoder(nn.Module):
    def __init__(self, model_name, model, pretrained):
        super().__init__()

        self.name = model_name

        self.global_params = model.global_params

        self.stem_conv = model._conv_stem
        self.stem_batch_norm = model._bn0
        self.stem_swish = Swish(name='stem_swish')
        self.blocks = model._blocks
        self.head_conv = model._conv_head
        self.head_batch_norm = model._bn1
        self.head_swish = Swish(name='head_swish')

    def forward(self, x):
        # Stem
        x = self.stem_conv(x)
        x = self.stem_batch_norm(x)
        x = self.stem_swish(x)

        # Blocks
        for idx, block in enumerate(self.blocks):
            drop_connect_rate = self.global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= idx / len(self.blocks)
            x = block(x, drop_connect_rate)

        # Head
        x = self.head_conv(x)
        x = self.head_batch_norm(x)
        x = self.head_swish(x)
        return x


class EfficientNetEncoder(nn.Module):
    def __init__(self, model_name='efficient-b1', pretrained=True):
        assert model_name.startswith('efficientnet-b')
        super().__init__()
        model = _get_model_by_name(model_name,  pretrained=pretrained)
        self.encoder = Encoder(model_name, model ,pretrained=pretrained)


    def forward(self, x):
        return list(get_blocks_to_be_concat(self.encoder, x).values())