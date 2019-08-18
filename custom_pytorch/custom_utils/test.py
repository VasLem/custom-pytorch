from custom_pytorch.custom_utils.data_prep import get_indices, get_sampler, get_loader
from custom_pytorch.custom_utils.epoch import TestEpoch
from torch import nn
import torch
from torch.utils.data import Dataset


class Tester:
    def __init__(self, dataset: Dataset, model: nn.Module, device='cuda'):
        self.dataset = dataset
        self.indices = get_indices(
            'test', config=None, dataset=self.dataset, weights=None)
        self.sampler = get_sampler('test', self.dataset, None, self.indices)
        self.loader = get_loader('test', self.dataset, self.sampler)
        self.sigmoid = nn.Sigmoid()
        self.epoch = TestEpoch(
            model, loss=None, metrics=None, device=device, verbose=True)

    def __call__(self):
        return self.epoch.run()

    def predict(self):
        return self.sigmoid(torch.from_numpy(self())).data.numpy()
