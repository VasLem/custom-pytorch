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
                weights: np.ndarray, indices: np.ndarray):
    """Get the sampler to be used in the dataloader

    :param stage: the stage
    :type stage: str
    :param config: the configuration
    :type config: Config
    :param dataset: the dataset
    :type dataset: Dataset
    :param weights: the total samples weights, with size equal to the number of the total samples
    :type weights: np.ndarray
    :param indices: the samples indices
    :type indices: np.ndarray
    :return: the sampler if the stage is not test, else None
    :rtype: SubsetRandomSampler|None
    """
    def valid_sampler():
        return SubsetRandomSampler(indices, replacement=config.train_replacement,
                                   weights=weights[indices]/np.sum(weights[indices]))

    def train_sampler(config: Config):
        def check_replacement(config, inds):
            if not config.train_replacement:
                if len(inds) > config.train_size:
                    print("The requested train size is too large for the existing dataset,"
                          "given that the samples are to be picked without replacement")
                    print("Setting train size to ", len(inds))
                    config.train_size = len(inds)
        if config.train_selection != 'any':
            if config.train_selection == 'all':
                np.random.shuffle(indices)
                print("Setting train size to", len(indices))
                config.train_size = len(indices)
                return SubsetRandomSampler(indices, replacement=False)
            elif config.train_selection == 'hi_weight':
                _indices = indices[weights[indices] == np.max(weights)]
                print('Number of highest weight indices found: ', len(_indices))
                check_replacement(config, _indices)
                return SubsetRandomSampler(_indices, replacement=config.train_replacement,
                                           num_samples=config.train_size,
                                           weights=weights[_indices]/np.sum(weights[_indices]))

        check_replacement(config, indices)
        return SubsetRandomSampler(indices, replacement=config.train_replacement,
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
