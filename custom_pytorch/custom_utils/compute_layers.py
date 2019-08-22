from math import log2, ceil
def compute_needed_layers(inp_channels, out_channels):
    """In order to get the input with inp_channels x to out_channels y, the assumption that is being
    made is that we will need, if x > y, n layers that can be retrieved from the following equation:
    (y + 1) ** n = x -> n = log(x) / log(y + 1)
    This assumption is constructed by principles of information theory.
    The target to a better architecture is to minimize the entropy of the input variable, such that
    the random variable can become deterministic, with an associated limit reaching infinity, without
    however losing generalization, which poses a regularization term to the computation of that limit.
    The original entropy of the input, given inp_channels, is:
    -sum(p(x) log(p(x))), where x is each channel, p is the generalized probability of the matrix x
    to have a particular state and the sum is performed over all channels. The layer induces a set
    of trainable parameters W, that can effectively reduce the entropy by a factor of the dimension
    of W . By using the general form of the layers, the entropy of the output of the first layer can be written
    as:
    -sum(p(nl(W*x)) * log(p(nl(W*x)))) = -sum(nl(Wp(x)) * log(nl(W * p(x)))
    The non-linearity does not permit any more solution steps, so it needs to be handled.
    Assuming, without loss of generality, that the following stands for a scalar x:
     nl(x) = a, x > c and nl(x) = b, otherwise, then for any input x, the output nl(x) will be
     a tensor that will include the elements a or b, after the comparison with c. This can
     be mathematically written using a step function:
     nl(x) = a * U(x - c) + b * (1 - U(x - c)), where U is the multidimensional step function.
     Then we have:
     nl(W * x) = a * U(W * x - c) + b * (1 - U(W * x - c)) = b + (a - b) * U(W * x - c)

    """
    if inp_channels > out_channels:
        ret = ceil(log2(inp_channels) / log2(out_channels + 1))
    else:
        ret = ceil(log2(out_channels) / log2(inp_channels + 1))
    return ret
