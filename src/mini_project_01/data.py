from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import torch


def load_mnist_dataset(data_dir: str = "data/",
                        threshold: int = 0.5,
                        batch_size: int= 128,
                        binarized: bool = True) -> tuple[DataLoader, DataLoader]:
    """
    Load the binarized MNIST dataset (train and test) from the `data_dir`.

    If the dataset is not found in the `data_dir`, it will be downloaded.

    Parameters:
    data_dir: [str]
        Directory where the dataset is stored.

    Returns:
    train_dataset: [torch.utils.data.Dataset]
        The training dataset.
    test_dataset: [torch.utils.data.Dataset]
        The test dataset.
    """

    # Load MNIST as binarized at 'thresshold' and create data loaders
    if binarized:
        mnist_train_loader = DataLoader(datasets.MNIST(data_dir, train=True, download=True,
                                                        transform=transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: (threshold < x).float().squeeze())])),
                                                        batch_size=batch_size, shuffle=True)
        mnist_test_loader = DataLoader(datasets.MNIST(data_dir, train=False, download=True,
                                                        transform=transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: (threshold < x).float().squeeze())])),
                                                        batch_size=batch_size, shuffle=True)
    else:
        transform = transforms.Compose([ transforms.ToTensor(),
                                        transforms.Lambda(lambda x : x + torch.rand(x.shape)/255),
                                        transforms.Lambda(lambda x : (x-0.5)*2.0),
                                        transforms.Lambda(lambda x : x.flatten())
                                        ])
        mnist_train_loader = DataLoader(datasets.MNIST(data_dir, train = True,
                                   download = True,
                                   transform = transform),batch_size=batch_size,shuffle=True)
        mnist_test_loader = DataLoader(datasets.MNIST(data_dir, train = False,
                                   download = True,
                                   transform = transform),batch_size=batch_size,shuffle=True)
    return mnist_train_loader, mnist_test_loader

    

if __name__ == "__main__":
    load_mnist_dataset()
