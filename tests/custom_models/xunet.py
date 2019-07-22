from custom_pytorch.custom_models.xunet import XUnet, _DecoderBlock, _Downsampler
from torch import nn
import torch
from torchvision.datasets import VOCSegmentation
from segmentation_models_pytorch.encoders import get_preprocessing_fn, get_encoder

class SimpleDecoderBlock:
    def __init__(self, inp_channels, out_channels, scale_ratio):
        sequence = []
        if scale_ratio > 1:
            sequence.append(nn.UpsamplingBilinear2d(scale_factor=scale_ratio))
        sequence.append(nn.Conv2d(inp_channels, out_channels))
        if scale_ratio < 1:
            sequence.append(nn.FractionalMaxPool2d(3, output_ratio=scale_ratio))
        sequence.append(nn.ReLU6)
        self.sequence = nn.Sequential(sequence)

    def forward(self, input):
        return self.sequence(input)

class SimpleDownsamplerBlock:
    def __init__(self, inp_channels, out_channels):
        self.sequence = nn.Sequential([
            nn.Conv2d(inp_channels, out_channels, 3),
            nn.ReLU6()])

    def forward(self, input):
        return self.sequence(input)


class SimpleXUnet(XUnet):
    def __init__(self, encoder_name, sample_input,  n_categories, shared_decoders=False):
        self.encoder = get_encoder(encoder_name, 'imagenet')
        inp_shape = sample_input.size()[-3:]
        feats = self.encoder(sample_input)
        out_shapes = [feat.size()[-3:] for feat in feats]
        super().__init__(inp_shape, SimpleDecoderBlock,
                 SimpleDownsamplerBlock,
                 out_shapes, shared_decoders=shared_decoders)

    def forward(self, inputs):
        enc_features = self.encoder(inputs)
        ret = super().forward(inputs, enc_features)

    def extract_features(self, input):
        enc_features = self.encoder(input)
        return super().extract_features(input, encoded_features=enc_features)

def main():
    encoder = 'resnet18'
    train_dataset = VOCSegmentation('/media/vaslem/Data/kaggle/input/VOCdevkit/VOC2012', image_set='train',
                                    transform=get_preprocessing_fn(encoder))
    valid_dataset = VOCSegmentation('/media/vaslem/Data/kaggle/input/VOCdevkit/VOC2012', image_set='val',
                                    transform=get_preprocessing_fn(encoder))
    data_loader = torch.utils.data.DataLoader(imagenet_data,
                                          batch_size=4,
                                          shuffle=True,
                                          num_workers=args.nThreads)


if __name__ == '__main__':
     main()









