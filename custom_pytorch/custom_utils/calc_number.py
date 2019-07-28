import numpy as np

def submodules_number(module):
    """Number of submodules of a given module

    :param module: the module
    :type module: nn.Module
    :return: the number of submodules
    :rtype: int
    """
    total_num = 0
    f = False
    for child in module.children():
        f = True
        total_num += submodules_number(child)
    if not f:
        return 1
    return total_num

def params_number(module):
    """Number of parameters of a given module

    :param module: the module
    :type module: nn.Module
    :return: the number of parameters
    :rtype: int
    """
    model_parameters = filter(lambda p: p.requires_grad, module.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params