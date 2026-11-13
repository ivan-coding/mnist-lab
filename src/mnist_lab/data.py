import torchvision
import torchvision.datasets as datasets

import numpy as np


def load_mnist():
    train_set = datasets.MNIST(root='./data', train=True, download=True,
                                           transform=torchvision.transforms.ToTensor())

    test_set = datasets.MNIST(root='./data', train=False, download=True,
                                           transform=torchvision.transforms.ToTensor())

    x_train = train_set.data.numpy().astype(np.float32) / 255.0
    y_train = train_set.targets.numpy().astype(np.int64)

    x_test = test_set.data.numpy().astype(np.float32) / 255.0
    y_test = test_set.targets.numpy().astype(np.int64)

    return x_train, y_train, x_test, y_test


if __name__ == "__main__":
    x_train, y_train, x_test, y_test = load_mnist()

    print(f"Train: {x_train.shape}, {y_train.shape}")
    print(f"Test:  {x_test.shape}, {y_test.shape}")