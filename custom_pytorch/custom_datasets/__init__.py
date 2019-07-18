import torch
import os
import torchvision
from torchvision import datasets

class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
    """

    def __init__(self, dataset, indices=None, num_samples=None):
        # if indices is not provided,
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices

        # if num_samples is not provided,
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples

        # distribution of classes in the dataset
        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1

        # weight for each sample
        weights = [1.0 / label_to_count[self._get_label(dataset, idx)]
                   for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, idx):
        dataset_type = type(dataset)
        if dataset_type is torchvision.datasets.MNIST:
            return dataset.train_labels[idx].item()
        elif dataset_type is torchvision.datasets.ImageFolder:
            return dataset.imgs[idx][1]
        else:
            try:
                return dataset.train_labels[idx].item()
            except AttributeError:
                try:
                    return dataset.imgs[idx][1]
                except BaseException:
                    raise NotImplementedError

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(
            self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples


class FilenamedImageFolder(datasets.ImageFolder):
    def __init__(self, *args, invalid_labels=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.invalid_labels = invalid_labels if invalid_labels is not None else []

    def __getitem__(self, index):
        filename = os.path.basename(self.imgs[index][0])
        im, label = super().__getitem__(index)
        if self.invalid_labels:
            if label in self.invalid_labels:
                label = -1
            else:
                for in_lab in self.invalid_labels:
                    if label >  in_lab:
                        label -= 1
        return  (im, label, filename)  # return image path
