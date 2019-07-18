def check_stage(stage, train=None, valid=None, test=None):
    if stage == 'train':
        return train
    if stage == 'valid':
        return valid
    if stage == 'test':
        return test
    raise ValueError(f"Unknown stage provided: {stage} ."
                     " It must be one of the following: 'train', 'valid', 'test'")