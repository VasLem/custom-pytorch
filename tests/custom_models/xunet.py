from custom_pytorch.custom_models.xunet import XUnet, _DecoderBlock, _Downsampler
from custom_pytorch.custom_config import Config
from torch import nn
import torch
from torch.utils.data import RandomSampler
from torch.optim import Adam
from torchvision.datasets import VOCSegmentation
from segmentation_models_pytorch.encoders import get_preprocessing_fn, get_encoder
from custom_pytorch.custom_visualizations.segmentation import Visualizer
from custom_pytorch.external.pytorch_enet.metric.iou import IoU as _IoU


class IoU(nn.Module):
    def __init__(self, n_categories):
        super().__init__()
        self.calculator = _IoU(n_categories)

    def forward(self, preds, gts):
        self.calculator.add(preds, gts)
        return self.calculator.value()[1]


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

    def forward(self, input):
        return self.sequence(input)

class SimpleDownsamplerBlock(_Downsampler):
    def __init__(self, inp_channels, out_channels):
        super().__init__(inp_channels, out_channels)
        self.sequence = nn.Sequential(
            nn.Conv2d(inp_channels, out_channels, 3, padding=1),
            nn.ReLU6())

    def forward(self, input):
        return self.sequence(input)


class SimpleXUnet(XUnet):
    def __init__(self, encoder_name, sample_input,  n_categories, shared_decoders=False, reversed_features=True):

        encoder = get_encoder(encoder_name, 'imagenet')

        in_model = nn.Conv2d(sample_input.size()[0], self.n_categories, 3, padding=1)
        sample_input = in_model(sample_input)
        inp_shape = sample_input.size()[-3:]
        feats = encoder(sample_input)
        if reversed_features:
            feats = feats[::-1]
        out_shapes = [feat.size()[-3:] for feat in feats]
        super().__init__(inp_shape, SimpleDecoderBlock,
                 SimpleDownsamplerBlock,
                 out_shapes, shared_decoders=shared_decoders)
        self.in_model = in_model
        self.reversed_features = reversed_features
        self.n_categories = n_categories
        self.out_model = nn.Conv2d(inp_shape[0], self.n_categories, 3, padding=1)
        self.encoder = encoder

    def forward(self, inputs):
        inputs = self.in_model(inputs)
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
    from tqdm import tqdm
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
        img[img == 255] = 0 # removing borders
        img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_NEAREST)
        img = torch.from_numpy(img).long()
        return img


    train_dataset = VOCSegmentation('/media/vaslem/Data/kaggle/input', image_set='train',
                                    transform=transform_func, target_transform=target_transform_func)
    valid_dataset = VOCSegmentation('/media/vaslem/Data/kaggle/input', image_set='val',
                                    transform=transform_func, target_transform=target_transform_func)

    CONFIG = Config(train_size=len(train_dataset), valid_size=100, batch_size=3,
                   random_seed=42, lr=1e-3, identifier='SimpleXUnet')
    train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                          batch_size=CONFIG.batch_size,
                                          shuffle=True,
                                          num_workers=8)
    valid_sampler = RandomSampler(valid_dataset, True, CONFIG.valid_size)
    valid_data_loader = torch.utils.data.DataLoader(valid_dataset, sampler=valid_sampler,
                                          batch_size=CONFIG.batch_size)
    device = 'cuda'
    model = SimpleXUnet(encoder, train_dataset[0][0].unsqueeze(dim=0), 21).to(device)
    for param in model.encoder.parameters():
        param.requires_grad = False
    # model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    # params = sum([np.prod(p.size()) for p in model_parameters])
    # print(params)
    optimizer = Adam(model.parameters(), CONFIG.lr)
    step = 0
    epochs_num = 30

    visualizer = Visualizer(CONFIG, metric_used='IoU', include_lr=False,
                            metric_function=IoU(21))
    loss_function = nn.CrossEntropyLoss(ignore_index=0)

    for epoch in range(epochs_num):
        print(f"Epoch {epoch + 1}")
        total_t_loss = 0
        total_v_loss = 0
        tb_cnt = 0
        t_metrics = []
        v_metrics = []
        for batch in tqdm(train_data_loader):

            optimizer.zero_grad()
            t_ims = batch[0].to(device)
            t_gts = batch[1].to(device)
            t_outs = model(t_ims)
            t_loss = loss_function(t_outs, t_gts)
            t_loss.backward()
            total_t_loss += t_loss.cpu().data.numpy()
            tb_cnt += 1
            optimizer.step()
            t_ims = t_ims.detach()
            t_gts = t_gts.detach()
            t_outs = t_outs.detach()
            if step % CONFIG.valid_every == 0:
                with torch.no_grad():
                    vb_cnt = 0
                    for v_batch in tqdm(valid_data_loader):
                        v_ims = v_batch[0].to(device)
                        v_gts = v_batch[1].to(device)
                        v_outs  = model(v_ims)
                        v_loss = loss_function(v_outs.detach(), v_gts.detach())
                        total_v_loss += v_loss.cpu().data.numpy()
                        vb_cnt += 1
            t_metric = visualizer.step(step, t_loss, t_ims, t_gts, t_outs, valid=False)
            v_metric = visualizer.step(step, v_loss, v_ims, v_gts, v_outs, valid=True)
            if t_metric is not None:
                t_metrics.append(t_metric)
            if v_metric is not None:
                v_metrics.append(v_metric)
            step += 1
        print("Training Loss: ", total_t_loss / tb_cnt)
        print("Validation Loss: ", total_v_loss / vb_cnt)
        print("Training IOU:", np.mean(t_metrics))
        print("Validation IOU:", np.mean(v_metrics))











if __name__ == '__main__':
     main()









