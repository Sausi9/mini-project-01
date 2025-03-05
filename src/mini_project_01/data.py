import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from typing import Tuple
import hydra
from helpers import VAE_CONSTS

with hydra.initialize(config_path="../../configs", version_base="1.3"):
    cfg = hydra.compose(config_name="config.yaml")

def logit_transform(x: torch.Tensor, alpha: float = 1e-5) -> torch.Tensor:
    """
    Apply a logit transform to data in (0,1). We first clamp/clip the data
    to [alpha, 1 - alpha], then apply logit.

    logit(x) = log(x / (1 - x))

    Parameters
    ----------
    x : torch.Tensor
        Input in the range [0,1].
    alpha : float
        Small constant for clipping to avoid infinite logit at 0 or 1.

    Returns
    -------
    torch.Tensor
        Logit-transformed tensor in (-∞, +∞).
    """
    # Clamp away from [0,1] boundaries
    x = torch.clamp(x, alpha, 1 - alpha)
    return torch.log(x) - torch.log1p(-x)


def dequant_noise(x: torch.Tensor) -> torch.Tensor:
    """
    Dequantize an 8-bit image by:
      1) Scaling x from [0,1] to [0,255]
      2) Adding uniform random noise U(0,1)
      3) Dividing by 256 to get back to [0,1]

    This gives a continuous value in [0,1] instead of discrete multiples of 1/256.

    Parameters
    ----------
    x : torch.Tensor in [0,1]
        Input image tensor.

    Returns
    -------
    torch.Tensor in [0,1]
        Dequantized image.
    """
    x = x * 255.0
    x = x + torch.rand_like(x)
    x = x / 256.0
    return x


def load_mnist_dataset(
    data_dir: str = "data/",
    batch_size: int = 128,
    binarized: bool = True,
    threshold: float = 0.5,
    flatten: bool = True,
    do_logit: bool = False,
    alpha: float = 1e-5,
    shuffle_test: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    """
    Load MNIST as either binarized or continuous, optionally dequantizing and
    logit-transforming the images. Returns train and test DataLoaders.

    Parameters
    ----------
    data_dir : str
        Directory where the MNIST dataset is (or will be) downloaded.
    batch_size : int
        Batch size for the DataLoaders.
    binarized : bool
        If True, load data in binarized form via 'threshold'.
    threshold : float
        Threshold for binarizing pixel values. If pixel > threshold => 1, else 0.
    flatten : bool
        If True, reshapes each image to (784,). Otherwise keeps shape [1, 28, 28].
    do_logit : bool
        If True, apply a logit transform after scaling/dequant.
        That is, x <- log(x/(1-x)).
    alpha : float
        Small constant for clipping during logit transform to avoid infinite values.
    shuffle_test : bool
        If True, shuffle the test set. Typically False is standard for reproducibility.

    Returns
    -------
    train_loader : DataLoader
        Training set DataLoader.
    test_loader : DataLoader
        Test set DataLoader.
    """

    # Build the base transform list
    base_transforms = [transforms.ToTensor()]  # Convert PIL image to [0,1] float

    if binarized:
        # If binarized, we threshold the data
        base_transforms.append(transforms.Lambda(lambda x: (x > threshold).float()))
    else:
        # If not binarized, do dequant
        base_transforms.append(transforms.Lambda(dequant_noise))

    # If we haven't binarized, the data is in [0,1]. Optionally do a logit transform.
    # Typically, we do it after dequantizing so that the logit is well-defined.
    if not binarized and do_logit:
        base_transforms.append(
            transforms.Lambda(lambda x: logit_transform(x, alpha=alpha))
        )

    # Flatten, if requested
    if flatten:
      if cfg.models.name == VAE_CONSTS:
        base_transforms.append(transforms.Lambda(lambda x: x.squeeze()))
      else:
        base_transforms.append(transforms.Lambda(lambda x: x.view(-1)))

    # Compose everything
    transform = transforms.Compose(base_transforms)

    # Build Datasets
    train_dataset = datasets.MNIST(
        root=data_dir, train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root=data_dir, train=False, download=True, transform=transform
    )

    # Build DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle_test)

    return train_loader, test_loader


if __name__ == "__main__":
    # Example usage:
    # 1) Binarized data
    bin_train, bin_test = load_mnist_dataset(
        binarized=True,
        threshold=0.5,
        flatten=True,
    )
    # 2) Continuous + dequant + logit
    cont_train, cont_test = load_mnist_dataset(
        binarized=False, do_logit=True, alpha=1e-5, flatten=True
    )

    # Check shapes
    x_batch, y_batch = next(iter(bin_train))
    print("[Binarized] x_batch shape:", x_batch.shape)
    x_batch, y_batch = next(iter(cont_train))
    print("[Dequant+Logit] x_batch shape:", x_batch.shape)
