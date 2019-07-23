from custom_pytorch.custom_models.xunet import XUnet, _DecoderBlock, _Downsampler
from custom_pytorch.custom_config import Config
from torch import nn
import torch
from torch.optim import Adam
from torchvision.datasets import VOCSegmentation
from segmentation_models_pytorch.encoders import get_preprocessing_fn, get_encoder
from custom_pytorch.custom_visualizations.segmentation import Visualizer
from custom_pytorch.external.pytorch_enet.metric.iou import IoU


class SimpleDecoderBlock(_DecoderBlock):
    def __init__(self, inp_channels, out_channels, scale_ratio):
        super().__init__(inp_channels, out_channels, scale_ratio)
        sequence = []
        if scale_ratio > 1:
            sequence.append(nn.UpsamplingBilinear2d(scale_factor=scale_ratio))
        sequence.append(nn.Conv2d(inp_channels, out_channels, 3))
        if scale_ratio < 1:
            sequence.append(nn.FractionalMaxPool2d(3, output_ratio=scale_ratio))
        sequence.append(nn.ReLU6(inplace=True))
        self.sequence = nn.Sequential(*sequence)

    def forward(self, input):
        return self.sequence(input)

class SimpleDownsamplerBlock(_Downsampler):
    def __init__(self, inp_channels, out_channels):
        super().__init__(inp_channels, out_channels)
        self.sequence = nn.Sequential(
            nn.Conv2d(inp_channels, out_channels, 3),
            nn.ReLU6())

    def forward(self, input):
        return self.sequence(input)


class SimpleXUnet(XUnet):
    def __init__(self, encoder_name, sample_input,  out_channels, shared_decoders=False, reversed_features=True):

        encoder = get_encoder(encoder_name, 'imagenet')
        inp_shape = sample_input.size()[-3:]
        self.reversed_features = reversed_features
        feats = encoder(sample_input)
        if self.reversed_features:
            feats = feats[::-1]
        out_shapes = [feat.size()[-3:] for feat in feats]
        out_channels = out_channels
        super().__init__(inp_shape, SimpleDecoderBlock,
                 SimpleDownsamplerBlock,
                 out_shapes, shared_decoders=shared_decoders)
        self.out_model = nn.Conv2d(inp_shape[0], self.out_channels, 3)
        self.encoder = encoder

    def forward(self, inputs):
        enc_features = self.encoder(inputs)
        if self.reversed_features:
            enc_features = enc_features[::-1]
        ret = super().forward(inputs, enc_features)
        return self.out_model(ret)

    def extract_features(self, input):
        enc_features = self.encoder(input)
        if self.reversed_features:
            enc_features = enc_features[::-1]
        return super().extract_features(input, encoded_features=enc_features)


def main():
    import tqdm
    import numpy as np
    from PIL import Image
    import cv2
    from torch import Tensor

    encoder = 'resnet18'
    def transform_func(img):
        img = np.array(img)
        img = get_preprocessing_fn(encoder)(img)
        img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
        img = np.transpose(img,(2,0,1))
        img = torch.from_numpy(img).float()

        return img

    def target_transform_func(img):
        img = np.array(img)
        img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
        img = torch.from_numpy(img).float()
        return img


    train_dataset = VOCSegmentation('/media/vaslem/Data/kaggle/input', image_set='train',
                                    transform=transform_func)
    valid_dataset = VOCSegmentation('/media/vaslem/Data/kaggle/input', image_set='val',
                                    transform=transform_func)

    CONFIG = Config(train_size=len(train_dataset), valid_size=len(valid_dataset), batch_size=8,
                   random_seed=42, lr=1e-3, identifier='SimpleXUnet')
    train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                          batch_size=CONFIG.batch_size,
                                          shuffle=True,
                                          num_workers=8)
    valid_data_loader = torch.utils.data.DataLoader(valid_dataset,
                                          batch_size=CONFIG.batch_size)
    model = SimpleXUnet(encoder, train_dataset[0][0].unsqueeze(dim=0), 1)
    optimizer = Adam(model.parameters(), CONFIG.lr)
    step = 0
    epochs_num = 30

    visualizer = Visualizer(CONFIG, metric_used='IoU', include_lr=False, metric_function=IoU(20))
    loss_function = nn.CrossEntropyLoss()
    for epoch in range(epochs_num):
        for batch in tqdm(train_data_loader):
            optimizer.zero_grad()
            t_ims = batch[0].cuda()
            t_gts = batch[1].cuda()
            t_outs = model(t_ims)

            t_loss = loss_function(t_outs, t_gts)
            t_ims = t_ims.cpu().data.numpy()
            t_gts = t_gts.cpu().data.numpy()
            t_outs = t_outs.cpu().data.numpy()
            t_loss.backward()
            optimizer.step()
            if step % CONFIG.valid_every == 0:
                with torch.no_grad():
                    for v_batch in tqdm(valid_data_loader):
                        v_ims = v_batch[0].cuda()
                        v_gts = v_batch[1].cuda()
                        v_outs  = model(v_ims)
                        v_loss = loss_function(v_outs, v_gts)
            visualizer.step(step, t_loss, t_ims, t_gts, t_outs, False)
            visualizer.step(step, v_loss, v_ims, v_gts, v_outs, True)
            step += 1










if __name__ == '__main__':
     main()









