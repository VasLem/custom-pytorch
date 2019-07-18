import torch
def where(cond, x_1, x_2):
    return (cond * x_1) + ((1-cond) * x_2)

def log_sigmoid(input):
    return where(input < 0, input, 0) - torch.log(torch.exp(-torch.abs(input)) + 1)
