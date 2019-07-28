from segmentation_models_pytorch.base.model import Model as _Model
from torch import nn

class Model(_Model):
    def initialize(self, model=None):
        obj = self
        if model is not None:
            obj = model
        for m in obj.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
