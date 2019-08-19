"""Helper functions. Their usage can be seen inside custom_utils.train.Trainer class
"""
import numpy as np
import torch
from numpy.random import choice
from torch.utils.data import Dataset

from custom_pytorch.custom_config import Config
from custom_pytorch.custom_samplers import SubsetRandomSampler
from custom_pytorch.custom_utils import check_stage


def get_indices(stage: str, config: Config, dataset: Dataset, weights: np.ndarray):
    """Return the indices of the samples to be used in the dataset provided

    :param stage: the stage, can be train, test or valid
    :type stage: str
    :param config: the configuration
    :type config: Config
    :param dataset: the dataset
    :type dataset: Dataset
    :param weights: the weights for the samples
    :type weights: np.ndarray
    :return: if stage != test, return a tuple (train_indices, valid_indices),
        else the vector of all the indices
    :rtype: np.ndarray|tuple(np.ndarray, np.ndarray)
    """
    if stage != 'test':
        valid_indices = choice(
            len(dataset), size=config.valid_size,
            replace=False, p=weights/np.sum(weights))
        train_indices = np.setdiff1d(range(len(dataset)), valid_indices)
        return train_indices, valid_indices
    else:
        return np.array(range(len(dataset)))


def get_sampler(stage: str, config: Config, dataset: Dataset,
                weights: np.ndarray, indices: np.ndarray, replacement: bool = False):
    """Get the sampler to be used in the dataloader

    :param stage: the stage
    :type stage: str
    :param config: the configuration
    :type config: Config
    :param dataset: the dataset
    :type dataset: Dataset
    :param weights: the samples weights
    :type weights: np.ndarray
    :param indices: the samples indices
    :type indices: np.ndarray
    :raises if: replacement is set to False and more samples are requested
    :return: the sampler if the stage is not test, else None
    :rtype: SubsetRandomSampler|None
    """
    def valid_sampler():
        if weights is None:
            return None
        return SubsetRandomSampler(indices, replacement=replacement,
                                   weights=weights[indices]/np.sum(weights[indices]))

    def train_sampler(config: Config):
        if weights is None:
            return None
        if config.train_size == 'all':
            np.random.shuffle(indices)
            config.train_size = len(indices)
            return SubsetRandomSampler(indices, replacement=False)
        # Negating replacement, so that more samples can be viewed per batch,
        #  will raise if more samples are requested
        return SubsetRandomSampler(indices, replacement=replacement,
                                   num_samples=config.train_size,
                                   weights=weights[indices]/np.sum(weights[indices]))

    def test_sampler():
        return None
    return check_stage(
        stage, train=train_sampler(config), test=test_sampler(), valid=valid_sampler()
    )


def get_loader(stage, config, dataset, sampler, collate_fn=None):
    return check_stage(
        stage, train=torch.utils.data.DataLoader(
            dataset, batch_size=config.batch_size, sampler=sampler,
            collate_fn=collate_fn, pin_memory=False,
            drop_last=False, timeout=0, worker_init_fn=None),
        valid=torch.utils.data.DataLoader(
            dataset, batch_size=config.valid_batch_size, sampler=sampler,
            collate_fn=collate_fn, pin_memory=False,
            drop_last=False, timeout=0, worker_init_fn=None),
        test=torch.utils.data.DataLoader(
            dataset, batch_size=config.batch_size, shuffle=False,
            collate_fn=collate_fn)
    )
