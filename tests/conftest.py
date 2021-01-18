from torch.utils.data import DataLoader
from pytest import fixture
from torchvision.datasets import MNIST
from torchvision.transforms import PILToTensor
import os


@fixture(scope="session")
def mnist_train_loader():
    os.makedirs("tests/datasets", exist_ok=True)
    loader = DataLoader(
        MNIST("tests/datasets", train=True, download=True, transform=PILToTensor()),
        batch_size=50,
        shuffle=True,
        num_workers=8,
    )
    return loader
