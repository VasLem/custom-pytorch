from torch.utils.data import DataLoader
from torch.optim import SGD

# from custom_pytorch.custom_layers import FrequencyLayer
from torch import nn


class SimpleConvNet(nn.Module):
    """
    convnet with 2 conv2D and 1 fc
    """

    def __init__(
        self,
        input_size,
        n_cat,
        filters_size=(3, 3),
        filters_kernels=(3, 3),
        final_maxpool=2,
    ):
        super().__init__()
        self.input_size = input_size
        self.n_cat = n_cat
        self.model = nn.Sequential(
            nn.Conv2d(1, filters_size[0], filters_kernels[0]),
            nn.ReLU(),
            nn.Conv2d(filters_size[0], filters_size[1], filters_kernels[1]),
            nn.ReLU(),
            nn.MaxPool2d(final_maxpool),
            nn.Flatten(),
            nn.Linear(
                (
                    (input_size - sum((x - 1) for x in filters_kernels))
                    // ((final_maxpool + 1))
                )
                ** 2
                * filters_size[-1],
                n_cat,
            ),
            nn.ReLU(),
        )

    def forward(self, inp):
        ret = self.model(inp.float())
        return ret


import torch

USE_CUDA = True
from sklearn.preprocessing import label_binarize


def test_mnist(mnist_train_loader: DataLoader):
    device = torch.device("cuda" if USE_CUDA else "cpu")
    net = SimpleConvNet(28, 10).to(device)
    ret = []
    labels = []

    optimizer = SGD(net.parameters(), 0.001)
    loss_function = nn.BCEWithLogitsLoss()
    for epoch in range(10):
        total_loss = 0
        for batch in mnist_train_loader:
            optimizer.zero_grad()
            imgs = batch[0].to(device)
            labels = (
                torch.from_numpy(label_binarize(batch[1], classes=range(10)))
                .float()
                .to(device)
            )
            ret = net(imgs)
            t_loss = loss_function(ret, labels)
            t_loss.backward()
            optimizer.step()
            total_loss += t_loss.data.cpu().numpy() / len(batch)
    print(total_loss)
    torch.save(
        {
            "model_state_dict": net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": total_loss,
        },
        "tests/mnist.pt",
    )
    return


# def test_frequency_layer(mnist_loader: DataLoader):

#     FrequencyLayer()
#     pass