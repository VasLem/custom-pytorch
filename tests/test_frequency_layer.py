from torch.utils.data import DataLoader
from torch.optim import SGD
from custom_pytorch.custom_layers.freq_layer import FFT

# from custom_pytorch.custom_layers import FrequencyLayer
from torch import nn
from copy import deepcopy as copy


class SimpleFrequencyNet(nn.Module):
    """
    frequency net with 1 frequency layer, 2 conv2d and 1 fc
    """

    def __init__(
        self,
        input_size,
        n_cat,
        filters_size=(6, 10),
        filters_kernels=(3, 5),
        final_maxpool=5,
    ):
        super().__init__()
        self.input_size = input_size
        self.n_cat = n_cat
        self.fft = FFT(input_size)
        self.model_real = nn.Sequential(
            nn.Conv2d(1, filters_size[0], filters_kernels[0]),
            nn.ReLU(),
            nn.Conv2d(filters_size[0], filters_size[1], filters_kernels[1]),
            nn.ReLU(),
            nn.MaxPool2d(final_maxpool),
            nn.Flatten(),
        )
        self.model_imag = copy(self.model_real)
        self.final_layer = nn.Linear(
            2
            * (
                (input_size - sum((x - 1) for x in filters_kernels))
                // ((final_maxpool))
            )
            ** 2
            * filters_size[-1],
            n_cat,
        )

    def forward(self, inp):
        inp = (inp / 255.0).float()
        freq = self.fft.apply(inp)
        imag_ret = self.model_real(freq[0].float())
        real_ret = self.model_imag(freq[1].float())
        ret = self.final_layer(torch.cat((imag_ret, real_ret), 1))

        return ret


class SimpleConvNet(nn.Module):
    """
    convnet with 2 conv2D and 1 fc
    """

    def __init__(
        self,
        input_size,
        n_cat,
        filters_size=(6, 10),
        filters_kernels=(3, 5),
        final_maxpool=5,
    ):
        super().__init__()
        self.input_size = input_size
        self.n_cat = n_cat
        self.model = nn.Sequential(
            nn.Conv2d(1, filters_size[0], filters_kernels[0]),
            nn.ReLU(),
            nn.Conv2d(filters_size[0], filters_size[1], filters_kernels[1]),
            nn.ReLU(),
            nn.MaxPool2d(final_maxpool),
            nn.Flatten(),
            nn.Linear(
                (
                    (input_size - sum((x - 1) for x in filters_kernels))
                    // ((final_maxpool))
                )
                ** 2
                * filters_size[-1],
                n_cat,
            ),
        )

    def forward(self, inp):
        ret = self.model(inp.float())
        return ret


import torch

USE_CUDA = True
from sklearn.preprocessing import label_binarize


def run_model(mnist_train_loader, model, step=0.01, optimizer=None):
    device = torch.device("cuda" if USE_CUDA else "cpu")
    ret = []
    labels = []
    model = model.to(device)
    if optimizer is None:
        optimizer = SGD(model.parameters(), step)
    loss_function = nn.BCEWithLogitsLoss()
    total_loss = 0
    for epoch in range(10):
        cnt = 0
        for batch in mnist_train_loader:
            optimizer.zero_grad()
            imgs = batch[0].to(device)
            labels = (
                torch.from_numpy(label_binarize(batch[1], classes=range(10)))
                .float()
                .to(device)
            )
            ret = model(imgs)
            t_loss = loss_function(ret, labels)
            t_loss.backward()
            optimizer.step()
            total_loss += t_loss.data.cpu().numpy() / len(batch)
            cnt += 1
        total_loss /= cnt - 1
    return total_loss


import os


def run_baseline_mnist_bce(mnist_train_loader: DataLoader, optimizer=None):
    try:
        with open("tmp_baseline_loss.txt", "r") as inp:
            loss = float(inp.read().rstrip().strip())
    except BaseException:
        loss = run_model(mnist_train_loader, SimpleConvNet(28, 10), optimizer=optimizer)
        with open("tmp_baseline_loss.txt", "w") as inp:
            inp.write(str(loss))
    return loss


def run_freq_net(mnist_train_loader: DataLoader, optimizer=None):
    loss = run_model(
        mnist_train_loader, SimpleFrequencyNet(28, 10), optimizer=optimizer
    )
    print(loss)
    return loss


def test_freq_net(mnist_train_loader):
    loss_compare = run_baseline_mnist_bce(mnist_train_loader)
    loss = run_freq_net(mnist_train_loader)
    print(loss_compare)
    print(loss)
