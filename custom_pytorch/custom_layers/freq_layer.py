from torch.autograd import Function
import numpy as np
from math import pi
from torch import from_numpy
from torch import nn
import torch

def init_mats(input_, reduce_dims=2):
    in_size = input_.shape
    dims = in_size[2:]
    real_mats = np.meshgrid(
        *(
            [np.arange(dim) / float(dim) for dim in dims]
            + [np.arange(dim / reduce_dims) for dim in dims]
        )
    )
    real_mat = np.cos(
        2 * pi * (real_mats[0] * real_mats[2] + real_mats[1] * real_mats[3])
    )
    imag_mat = np.sin(
        -2 * pi * (real_mats[0] * real_mats[2] + real_mats[1] * real_mats[3])
    )
    real_mat = from_numpy(real_mat)
    imag_mat = from_numpy(imag_mat)
    return real_mat, imag_mat


class FFT(Function):

    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(
        ctx,
        _input: torch.Tensor,
        reduce_dims=2,
    ):
        try:
            (
                real_mat,
                imag_mat,
            ) = ctx.saved_tensors
        except ValueError:
            real_mat, imag_mat = init_mats(_input, reduce_dims)
            ctx.save_for_backward(real_mat, imag_mat)
        real_mat = real_mat.to(_input.device)
        imag_mat = imag_mat.to(_input.device)

        real_output = (
            real_mat[np.newaxis, np.newaxis, ...] * _input[..., np.newaxis, np.newaxis]
        ).sum(axis=(2, 3))
        imag_output = (
            imag_mat[np.newaxis, np.newaxis, ...] * _input[..., np.newaxis, np.newaxis]
        ).sum(axis=(2, 3))
        return (real_output, imag_output)  # pytorch does not support complex numbers


class FrequencyLayer(nn.Module):
    """
    Returns : A Module that can be used
        inside nn.Sequential
    """

    def __init__(self, input_size):
        super().__init__()
        self.oper = FFT(input_size)

    def forward(self, x):
        return self.oper(x)
