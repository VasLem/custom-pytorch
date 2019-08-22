from custom_pytorch.custom_utils.data_prep import get_indices, get_sampler, get_loader,\
    get_tta_collate_fn
from custom_pytorch.custom_datasets import get_tta_dataset
from custom_pytorch.custom_utils.epoch import TestEpoch
from custom_pytorch.custom_utils.train import Trainer
from torch import nn
import torch
from torch.utils.data import Dataset
from custom_pytorch.custom_config import Config




class Tester:
    def __init__(
            self, config: Config, dataset: Dataset, inp_index,
            model: nn.Module, device='cuda', collate_fn=None,
            tta_functions=None, inv_tta_functions=None):
        self.dataset = dataset
        config.tta_functions = tta_functions
        config.inv_tta_functions = inv_tta_functions
        if config.apply_tta:
            self.dataset = get_tta_dataset(dataset, config.tta_number, config.tta_functions)
            collate_fn = get_tta_collate_fn('test', config, collate_fn)
        self.indices = get_indices(
            'test', config=None, dataset=self.dataset, weights=None)
        self.sampler = get_sampler('test', self.dataset, None, self.indices)
        self.loader = get_loader('test', self.dataset, self.sampler)
        self.sigmoid = nn.Sigmoid()
        self.epoch = TestEpoch(
            model, loss=None, metrics=None, device=device, verbose=True)
        self.step = lambda logs: self.epoch.run(
            self.loader, inp_index, inp_index, _logs=logs, _tta=config.apply_tta,
             _tta_strategy=config.tta_strategy, _tta_inv_transforms=inv_tta_functions)

    def __call__(self, _logs=None):
        return self.step(_logs)

    def predict(self, _logs=None):
        return self.sigmoid(torch.from_numpy(self(_logs))).data.numpy()
