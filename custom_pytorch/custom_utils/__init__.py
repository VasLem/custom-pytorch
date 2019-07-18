from custom_pytorch.custom_config import Config
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

def get_model_name(config: Config, epoch=None, train_loss=None, valid_loss=None):
    """Returns the current model snapshot name, given the current configuration, epoch and losses

    :param config: the configuration
    :type config: Config
    :param epoch: the epoch, defaults to None
    :type epoch: int, optional
    :param train_loss: the training loss, defaults to None
    :type train_loss: float, optional
    :param valid_loss: the validation loss, defaults to None
    :type valid_loss: float, optional
    :return: the model snapshot name
    :rtype: str
    """
    name = f'{config.identifier}_D_{config.date}_DS_{config.train_size}'.replace(' ', '_')

    if epoch is not None or train_loss is not None or valid_loss is not None:
        name += "_Ep_%d_TL_%.2f_VL_%.2f.pkl"%(epoch, train_loss, valid_loss)

    return name
