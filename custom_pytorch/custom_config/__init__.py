import math
from datetime import datetime
from easydict import EasyDict


class Config:
    def __init__(self, train_size=None,
                 train_selection='any',
                 train_replacement=False,
                 valid_size=None,
                 batch_size=None,
                 apply_tta=False,
                 tta_samples='auto',
                 tta_strategy='mean',
                 input_index=None,
                 gt_index=None,
                 random_seed=None,
                 lr=None, identifier='', valid_every=50, plot_train_every_n_steps=100,
                 plot_valid_every_n_steps=100, show_examples_every_n_steps=50, loss_name=None,
                 metrics_names=None, valid_indices=None, date=None,
                 **kwargs):
        """Main Configuration for every experiment

        :param train_size: training data size, defaults to None
        :type train_size: int, optional
        :param train_selection: the way to pick train samples, accepted values are:
            - any: Pick any samples from the ones left out of the validation
            - all: Pick all training samples available once,
                train_size and train_replacement is ignored
            - hi_weight: Pick only the samples with the highest weight
            Defaults to `any`
        :type train_selection: str
        :param train_replacement: whether to pick the training samples
            using replacement, defaults to False
        :type train_replacement: bool
        :param valid_size: validation data size, defaults to None
        :type valid_size: int, optional
        :param batch_size: batch size, defaults to None
        :type batch_size: int, optional
        :param apply_tta: whether to apply test time augmentation
        :type apply_tta: bool
        :param tta_samples: the number of samples to use in TTA. Can be integer or `auto`, where
            the optimal number will be found using validation samples.
        :type tta_samples: int|str
        :param tta_strategy: the TTA strategy to follow, can be `min` or `mean`, defaults to `mean`
        :type tta_strategy: str
        :param input_index: the index of the dataset batch to use as input
        :type input_index: str|int
        :param gt_index: the index of the dataset batch to use as ground truth
        :type gt_index: str|int
        :param random_seed: random seed, defaults to None
        :type random_seed: inr, optional
        :param lr: learning rate, defaults to None
        :type lr: float, optional
        :param identifier: the experiment source identifier, defaults to ``
        :type identifier: str, optional
        :param loss_name: the loss being used
        :type loss_name: str
        :param metrics_names: the list of metrics being used
        :type metrics_names: list(str)
        :param valid_indices: the samples indices to be used for validation
        :type valid_indices: numpy.ndarray
        :param date: the date of the experiment, if not provided it will
            be filled with current date string
        :type date: str
        """
        assert identifier, 'Identifier must not be empty'
        self.valid_every = valid_every
        self.plot_train_every_n_steps = plot_train_every_n_steps
        self.plot_valid_every_n_steps = plot_valid_every_n_steps
        self.show_examples_every_n_steps = show_examples_every_n_steps
        self.train_size = train_size
        self.train_selection = train_selection
        self.train_replacement = train_replacement
        self.valid_size = valid_size
        self.batch_size = batch_size
        self.random_seed = random_seed
        self.lr = lr
        self.identifier = identifier
        self.loss_name = loss_name
        self.metrics_names = metrics_names
        self.tta_samples = tta_samples
        self.tta_strategy = tta_strategy
        self.apply_tta = apply_tta
        self.input_index = input_index
        self.gt_index = gt_index
        self.valid_indices = valid_indices
        if date is None:
            self.date = str(datetime.now())
        for kwarg in kwargs:
            if isinstance(kwargs[kwarg], dict):
                setattr(self, kwarg, EasyDict(kwargs[kwarg]))
            else:
                setattr(self, kwarg, kwargs[kwarg])

    @property
    def valid_batches_number(self):
        """
        :return: The number of batches in validation
        :rtype: int
        """
        return math.ceil(self.valid_size / self.batch_size)

    @property
    def train_batches_number(self):
        """
        :return: The number of batches in training
        :rtype: int
        """
        return math.ceil(self.train_size / self.batch_size)
