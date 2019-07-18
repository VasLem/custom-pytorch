def check_stage(stage, train=None, valid=None, test=None):
    if stage == 'train':
        return train
    if stage == 'valid':
        return valid
    if stage == 'test':
        return test
    raise ValueError(f"Unknown stage provided: {stage} ."
                     " It must be one of the following: 'train', 'valid', 'test'")

def apply_reduction(value, reduction):
    """reduction emulator

    :param value: the tensor
    :type value: Tensor
    :param reduction: mean, sum, max or none
    :type reduction: str
    """
    if reduction == 'mean':
        return value.mean()
    if reduction == 'sum':
        return value.sum()
    if reduction != 'none':
        raise ValueError("Unknown reduction mode given, mean, sum or none expected")
