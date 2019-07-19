import math
from datetime import datetime
from easydict import EasyDict


class Config:
    def __init__(self, train_size=None,
                 valid_size=None,
                 batch_size=None, random_seed=None,
                 lr=None, identifier='', date=None, **kwargs):
        """Main Configuration for every experiment

        :param train_size: training data size, defaults to None
        :type train_size: int, optional
        :param valid_size: validation data size, defaults to None
        :type valid_size: int, optional
        :param batch_size: batch size, defaults to None
        :type batch_size: int, optional
        :param random_seed: random seed, defaults to None
        :type random_seed: inr, optional
        :param lr: learning rate, defaults to None
        :type lr: float, optional
        :param identifier: the experiment source identifier, defaults to ''
        :type identifier: str, optional
        :param date: the date of the experiment, if not provided it will
            be filled with current date string
        :type date: str
        """
        self.train_size = train_size
        self.valid_size = valid_size
        self.batch_size = batch_size
        self.random_seed = random_seed
        self.lr = lr
        self.identifier = identifier
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
