import os
from torchvision import datasets
from torch.utils.data import Dataset


def get_tta_dataset(test_dataset: Dataset, tta_number: int, tta_transforms):
    """Override supplied dataset __getitem__ method, to retrieve `tta_number` times the same sample,
    as a list. Applies the TTA tranformations as provided.

    :param test_dataset: the dataset to override
    :type test_dataset: Dataset
    :param tta_number: the number of times the same sample will be retrieved
    :type tta_number: int
    :param tta_transforms: the transforms, of size `tta_number`
    :type tta_transforms: list
    :return: the overriden dataset
    :rtype: Dataset
    """
    try:
        test_dataset.ntta_getitem
    except AttributeError:
        test_dataset.ntta_getitem = test_dataset.__getitem__

    def apply_tta(self, index):
        ret = []
        for cnt in range(tta_number):
            ret.append(tta_transforms[cnt](test_dataset.ntta_getitem[index]))
        return ret
    test_dataset.__getitem__ = apply_tta


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
                    if label > in_lab:
                        label -= 1
        return (im, label, filename)  # return image path
