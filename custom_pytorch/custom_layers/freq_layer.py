from torch.autograd import Function
import numpy as np
from math import pi
from torch import from_numpy
from collections import Iterable
from torch import nn, eye
from torch.functional import Tensor


def init_mats(input_):
    in_size = input_.shape
    dims = in_size[2:]
    real_mats = np.meshgrid(
        *(
            [np.arange(dim) / float(dim) for dim in dims]
            + [np.arange(dim) for dim in dims]
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
        ctx, _input, real_weight=None, imag_weight=None, real_bias=None, imag_bias=None
    ):
        try:
            (
                _,
                real_weight,
                imag_weight,
                real_bias,
                imag_bias,
                real_mat,
                imag_mat,
            ) = ctx.saved_tensors
        except ValueError:
            if real_weight is None:
                real_weight = eye(_input.shape[2])
            else:
                assert isinstance(real_weight, Tensor)
            if imag_weight is None:
                imag_weight = eye(_input.shape[2])
            else:
                assert isinstance(imag_weight, Tensor)
            if real_bias is not None:
                assert isinstance(real_bias, Tensor)
            if imag_bias is not None:
                assert isinstance(imag_bias, Tensor)
            real_mat, imag_mat = init_mats(_input)
        real_mat = real_mat.to(_input.get_device())
        imag_mat = imag_mat.to(_input.get_device())
        ctx.save_for_backward(
            _input, real_weight, imag_weight, real_bias, imag_bias, real_mat, imag_mat
        )
        real_output = (
            real_mat[np.newaxis, np.newaxis, ...] * _input[..., np.newaxis, np.newaxis]
        ).sum(axis=(2, 3))
        imag_output = (
            imag_mat[np.newaxis, np.newaxis, ...] * _input[..., np.newaxis, np.newaxis]
        ).sum(axis=(2, 3))
        return (real_output, imag_output)  # pytorch does not support complex numbers

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        (
            _input,
            real_weight,
            imag_weight,
            real_bias,
            imag_bias,
            real_mat,
            imag_mat,
        ) = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(real_weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(_input.mul(real_mat))
        if real_bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)

        return grad_input, grad_weight, grad_bias


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
