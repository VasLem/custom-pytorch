from custom_pytorch.custom_models import SimpleXUnetV2, SimpleXUnet
from custom_pytorch.custom_config import Config
from torch import nn
import torch
from torch.utils.data import RandomSampler
from torch.optim import SGD
from torchvision.datasets import VOCSegmentation
from segmentation_models_pytorch.encoders import get_preprocessing_fn, get_encoder
import segmentation_models_pytorch as smp
from custom_pytorch.custom_utils.calc_number import params_number, submodules_number
from custom_pytorch.custom_visualizations.segmentation import Visualizer
from custom_pytorch.external.pytorch_enet.metric.iou import IoU as _IoU
import torch.nn.functional as F


class IoU(nn.Module):
    def __init__(self, n_categories):
        super().__init__()
        self.calculator = _IoU(n_categories)

    def forward(self, preds, gts):
        self.calculator.add(preds, gts)
        return self.calculator.value()[1]

from efficientunet import *



def main():
    from tqdm import tqdm
    import numpy as np
    from PIL import Image
    import cv2
    from torch import Tensor
    from custom_pytorch.models.efficient_net_encoder import EfficientNetEncoder




    # encoder = EfficientNetEncoder('efficientnet-b3', pretrained=True).to(DEVICE)
    # encoder = 'resnet18'
    encoder = 'se_resnext50_32x4d'
    def transform_func(img):
        img = np.array(img)
        img = get_preprocessing_fn(encoder)(img)
        img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_LINEAR)
        img = np.transpose(img,(2,0,1))
        img = torch.from_numpy(img).float()

        return img

    def target_transform_func(img):
        img = np.array(img)
        img[img == 255] = 0 # removing borders
        img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_NEAREST)
        img = torch.from_numpy(img).long()
        return img


    train_dataset = VOCSegmentation('/media/vaslem/Data/kaggle/input', image_set='train',
                                    transform=transform_func, target_transform=target_transform_func)
    valid_dataset = VOCSegmentation('/media/vaslem/Data/kaggle/input', image_set='val',
                                    transform=transform_func, target_transform=target_transform_func)

    model_to_use = 'SimpleXUnetV2'
    # model_to_use = 'Unet'
    CONFIG = Config(train_size=len(train_dataset), valid_size=100, batch_size=3,
                   random_seed=42, lr=1e-2, identifier=model_to_use)
    train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                          batch_size=CONFIG.batch_size,
                                          shuffle=True,
                                          num_workers=8)
    valid_sampler = RandomSampler(valid_dataset, True, CONFIG.valid_size)
    valid_data_loader = torch.utils.data.DataLoader(valid_dataset, sampler=valid_sampler,
                                          batch_size=CONFIG.batch_size)
    device = 'cuda'
    if model_to_use == 'SimpleXUnet':

        model = SimpleXUnet(encoder, train_dataset[0][0].unsqueeze(dim=0), 21).to(device)
    elif model_to_use == 'SimpleXUnetV2':

        model = SimpleXUnetV2(encoder, train_dataset[0][0].unsqueeze(dim=0), 21).to(device)

    elif model_to_use == 'Unet':
        model = smp.Unet(encoder, classes=21).to(device)
    print("Model parameters number:", params_number(model))
    print("Model modules number:", submodules_number(model))
    for param in model.encoder.parameters():
            param.requires_grad = False
    # model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    # params = sum([np.prod(p.size()) for p in model_parameters])
    # print(params)
    optimizer = SGD(model.parameters(), CONFIG.lr)
    step = 0
    epochs_num = 30

    visualizer = Visualizer(CONFIG, metric_used='IoU', include_lr=False,
                            metric_function=IoU(21))
    loss_function = nn.CrossEntropyLoss()

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









