"""Herein resides Trainer, a useful class to be used while training, better in overriden fashion
"""

from .data_prep import get_indices, get_sampler, get_loader
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset

from custom_pytorch.custom_utils.epoch import TrainEpoch, ValidEpoch
from custom_pytorch.custom_config import Config
from custom_pytorch.custom_logs import Logger


class Trainer:

    def __init__(self, config: Config, train_dataset: Dataset, valid_dataset: Dataset,
                 collate_fn: callable,
                 model: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 loss_function: callable,
                 metric_functions: List[callable],
                 device='cuda', verbose=True,
                 samples_weights: np.ndarray = None):
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.train_loss_logs = {}
        self.valid_loss_logs = {}
        self.main_logger = Logger(config, 'logs', create_dir=True)

        self.weights = samples_weights
        if self.weights is None:
            self.weights = np.ones(len(train_dataset))
        self.train_indices, self.valid_indices = get_indices(
            'train', self.train_dataset, self.weights)
        self.train_sampler = get_sampler(
            'train', self.train_dataset, self.weights, self.train_indices)
        self.valid_sampler = get_sampler(
            'valid', self.valid_dataset, self.weights, self.valid_indices)
        self.train_loader = get_loader(
            'train', self.train_dataset, self.train_sampler, collate_fn=collate_fn)
        self.valid_loader = get_loader(
            'valid', self.valid_dataset, self.valid_sampler, collate_fn=collate_fn)
        self.train_epoch = TrainEpoch(model, loss_function,
                                      metric_functions, optimizer, device, verbose)
        self.valid_epoch = ValidEpoch(
            model, loss_function, metric_functions, device, verbose)
        self.train_step = lambda logs: self.train_epoch.run(
            self.train_loader, _logs=logs)
        self.valid_step = lambda logs: self.valid_epoch.run(
            self.valid_loader, _logs=logs)
        self.step = lambda logs, valid: self.train_step(
            logs) if not valid else self.valid_step(logs)
        self.train_logs = {}
        self.valid_logs = {}

    def write_logs(self, step, step_logs, valid):
        logs = self.train_logs
        if valid:
            logs = self.valid_logs
            self.valid_loss_logs = {}
        else:
            self.train_loss_logs = {}
        logs[step] = {'main logs': step_logs}
        self.main_logger.update(
            step=step, logs=logs[step]['main logs'], valid=valid)

    @property
    def optimizer(self):
        return self.train_epoch.optimizer

    @optimizer.setter
    def optimizer(self, value):
        self.train_epoch.optimizer = value
