from torch.autograd import Function
import numpy as np
from math import pi
from torch import from_numpy

class FrequencyLayer(Function):

    def __new__(cls, in_size, out_size):
        dims = in_size[1:]
        if len(dims) == 1:
            cls.real_mat = np.cos(2 * pi * np.arange(dims[0])/ float(dims[0]))
        else:
            cls.real_mat = np.meshgrid(tuple([np.cos(-2 * pi * np.arange(dim)/ float(dim))
                                         for dim in dims]))
        if len(dims) == 1:
            cls.imag_mat = np.sin(2 * pi * np.arange(dims[0])/ float(dims[0]))
        else:
            cls.imag_mat = np.meshgrid(tuple([np.sin(-2 * pi * np.arange(dim)/ float(dim))
                                         for dim in dims]))
        cls.real_mat = from_numpy(cls.real_mat)
        cls.image_mat = from_numpy(cls.imag_mat)
        cls.real_mat = cls.real_mat.view(*list((1, 1) + cls.real_mat.shape))
        cls.imag_mat = cls.imag_mat.view(*list((1, 1) + cls.imag_mat.shape))
        return Function.__new__(cls)


    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, _input, weight, bias=None):
        ctx.save_for_backward(_input, weight, bias)
        real_output = _input.mm((weight.mul(ctx.real_mat)).t())
        imag_output = _input.mm((weight.mul(ctx.imag_mat)).t())
        if bias is not None:
            real_output += bias[0].unsqueeze(0).expand_as(real_output)
            imag_output += bias[1].unsqueeze(0).expand_as(imag_output)


        return (real_output, imag_output) # pytorch does not support complex numbers

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        _input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(_input.mul(ctx.real_mat))
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)

        return grad_input, grad_weight, grad_bias