from custom_pytorch.custom_utils.data_prep import get_indices, get_sampler, get_loader
from custom_pytorch.custom_utils.epoch import TestEpoch
from torch import nn
import torch
from torch.utils.data import Dataset


class Tester:
    def __init__(self, dataset: Dataset, inp_index, model: nn.Module, device='cuda'):
        self.dataset = dataset
        self.indices = get_indices(
            'test', config=None, dataset=self.dataset, weights=None)
        self.sampler = get_sampler('test', self.dataset, None, self.indices)
        self.loader = get_loader('test', self.dataset, self.sampler)
        self.sigmoid = nn.Sigmoid()
        self.epoch = TestEpoch(
            model, loss=None, metrics=None, device=device, verbose=True)
        self.step = lambda logs: self.epoch.run(
            self.loader, inp_index, inp_index, _logs=logs)

    def __call__(self, _logs=None):
        return self.step(_logs)

    def predict(self, _logs=None):
        return self.sigmoid(torch.from_numpy(self(_logs))).data.numpy()
