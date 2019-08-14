import torch
from torch import nn
from torch.autograd import Function
from numpy.random import random


class Clipper(Function):
    """Clips computed weight during backward pass between provided boundaries
    """

    @staticmethod
    def forward(ctx, inputs, weight, min_bound, max_bound):
        ctx.save_for_backward(inputs, weight)
        ctx.min_bound = min_bound
        ctx.max_bound = max_bound
        return inputs

    @staticmethod
    def backward(ctx, grad_output):
        inputs, weight = ctx.saved_tensors
        weight.grad = torch.min(
            ctx.max_bound, torch.max(ctx.min_bound, weight.grad + weight)) - weight
        return grad_output


class ProductConv2D(nn.Module):

    r"""
    Modified Conv2d layer that performs the following operation:
    y = {\prod}_i{x_i^{w_i}}
    This operation is highly unstable, implying that gradients can easily vanish or explode.
    To remedy this, we set a constraint to the gradients to reside between the range [-1, 1]
    """
    MIN_BOUND = -1
    MAX_BOUND = 1

    def __init__(self, input_features, output_features, kernel_size, **kwargs):
        super().__init__()
        self.conv_layer = nn.Conv2d(
            input_features, output_features, kernel_size, **kwargs)

    def forward(self, inputs):
        out = self.conv_layer(torch.log(inputs))
        out = torch.exp(out)
        return Clipper.apply(out, self.conv_layer.weight, self.MIN_BOUND, self.MAX_BOUND)


def main():
    net = nn.Sequential(ProductConv2D(1, 3, 3, padding=1))
    sample = torch.from_numpy(random((1, 1, 64, 64))).float()
    expected_output = torch.from_numpy(random((1, 3, 64, 64))).float()
    optim = torch.optim.SGD(net.parameters(), lr=0.001)
    loss_func = nn.MSELoss()
    optim.zero_grad()
    out = net(sample)
    print(out.size())
    loss = loss_func(out, expected_output)
    loss.backward()
    optim.step()


if __name__ == '__main__':
    main()
