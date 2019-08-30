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
from custom_pytorch.custom_snapshots import SnapshotsHandler


class Trainer:

    def __init__(self, config: Config, train_dataset: Dataset, valid_dataset: Dataset,
                 inp_index, gt_index,
                 collate_fn: callable,
                 model: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 loss_function: callable,
                 metric_functions: List[callable],
                 device='cuda', verbose=True,
                 samples_weights: np.ndarray = None):
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.main_logger = Logger(config, 'logs', create_dir=True)
        self.config = config
        self.config.input_index = inp_index
        self.config.gt_index = gt_index
        config.metrics_names = [f.__name__ for f in metric_functions]
        config.loss_name = loss_function.__name__
        self.weights = samples_weights
        if self.weights is None:
            self.weights = np.ones(len(train_dataset))
        self.train_indices, self.valid_indices = get_indices(
            'train', config, self.train_dataset, self.weights)
        self.train_sampler = get_sampler(
            'train', config, self.train_dataset, self.weights, self.train_indices)
        self.valid_sampler = get_sampler(
            'valid', config, self.valid_dataset, self.weights, self.valid_indices)
        self.train_loader = get_loader(
            'train', config, self.train_dataset, self.train_sampler, collate_fn=collate_fn)
        self.valid_loader = get_loader(
            'valid', config, self.valid_dataset, self.valid_sampler, collate_fn=collate_fn)
        self.train_epoch = TrainEpoch(model, loss_function,
                                      metric_functions, optimizer, device, verbose)
        self.valid_epoch = ValidEpoch(
            model, loss_function, metric_functions, device, verbose)
        self.train_step = lambda logs: self.train_epoch.run(
            self.train_loader, inp_index, gt_index, _logs=logs)
        self.valid_step = lambda logs, _tta=False, _tta_strategy='mean',\
            _tta_inv_transforms=None: self.valid_epoch.run(
                self.valid_loader, inp_index, gt_index, _logs=logs, _tta=_tta,
                _tta_strategy=_tta_strategy, _tta_inv_transforms=_tta_inv_transforms)
        self.step = lambda logs, valid: self.train_step(
            logs) if not valid else self.valid_step(logs)
        self.snasphots_handler = SnapshotsHandler(
            self, 'snapshots', create_dir=True)
        self.train_logs = {}
        self.valid_logs = {}

    def write_logs(self, step_logs, valid):
        step = self.epoch
        logs = self.train_logs
        if valid:
            logs = self.valid_logs
        logs[step] = {'main logs': step_logs}
        self.main_logger.update(
            step=step, logs=logs[step]['main logs'], valid=valid)

    def load_model(self, model_name, **params):
        self.snasphots_handler.load(model_name, **params)
        self.config.warm_start = model_name
        return self.model

    def save_last_model(self, train_loss, valid_loss, train_metric, valid_metric):
        self.snasphots_handler.save(
            self.epoch, train_loss, valid_loss, train_metric, valid_metric, id='last')

    def save_best_model(self, train_loss, valid_loss, train_metric, valid_metric):
        self.snasphots_handler.save(
            self.epoch, train_loss, valid_loss, train_metric, valid_metric, id='best')

    @property
    def epoch(self):
        return self.train_epoch.epoch

    @epoch.setter
    def epoch(self, epoch):
        self.train_epoch.epoch = epoch
        self.valid_epoch.epoch = epoch

    @property
    def model(self):
        return self.train_epoch.model

    @model.setter
    def model(self, model):
        self.train_epoch.model = model
        self.valid_epoch.model = model

    @property
    def optimizer(self):
        return self.train_epoch.optimizer

    @optimizer.setter
    def optimizer(self, value):
        self.train_epoch.optimizer = value
