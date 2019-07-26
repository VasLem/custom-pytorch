import torch
from torch.utils.data import Sampler
from numpy.random import choice
class SubsetRandomSampler(Sampler):
    """Samples elements randomly from a given list of indices, with or without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """
    def __init__(self, indices, replacement=False, num_samples=None, p=None):
        self.replacement = replacement
        self.indices = indices
        self._num_samples = num_samples
        self.p = p


        if not isinstance(self.replacement, bool):
            raise ValueError("replacement should be a boolean value, but got "
                             "replacement={}".format(self.replacement))

        if self._num_samples is not None and not replacement:
            raise ValueError("With replacement=False, num_samples should not be specified, "
                             "since a random permute will be performed.")

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError("num_samples should be a positive integer "
                             "value, but got num_samples={}".format(self.num_samples))

    @property
    def num_samples(self):
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.indices)
        return self._num_samples


    def __iter__(self):
        indices = choice(self.indices, self.num_samples, replace=self.replacement, p=self.p)
        for ind in indices:
            yield ind

    def __len__(self):
        return self.num_samples