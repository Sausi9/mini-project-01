import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List
import hydra


# Load configuration
with hydra.initialize(config_path="../../../configs", version_base="1.3"):
    cfg = hydra.compose(config_name="config.yaml")



class InvertiblePermutation(nn.Module):
    """
    An invertible, fixed (random) permutation layer for 1D data of shape (B, D).

    This layer permutes the order of features, which helps ensure that different
    features are masked in different ways across coupling layers.
    """

    def __init__(self, num_features: int, seed: int = 42):
        """
        Parameters
        ----------
        num_features : int
            Dimensionality of the feature space.
        seed : int
            Random seed for generating the permutation.
        """
        super().__init__()
        # Create a fixed random permutation of [0, 1, 2, ..., num_features-1]
        rng = np.random.RandomState(seed)
        perm = np.arange(num_features)
        rng.shuffle(perm)

        # We store both the forward permutation and its inverse
        self.register_buffer("perm", torch.from_numpy(perm))
        
        # argsort to get inverse permutation
        inv = np.argsort(perm)
        self.register_buffer("inv_perm", torch.from_numpy(inv))

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Permute the features of x.

        Parameters
        ----------
        x : torch.Tensor of shape (B, D)
            Input data (features).

        Returns
        -------
        x_perm : torch.Tensor of shape (B, D)
            Permuted data.
        log_det : torch.Tensor of shape (B,)
            The log-determinant of the Jacobian for this transform.
            For a permutation, it is always 0, so we return a zero tensor.
        """
        # Note: permutation does not change volume, so log_det = 0
        x_perm = x[:, self.perm]
        batch_size = x.shape[0]
        log_det = x.new_zeros(batch_size)
        return x_perm, log_det

    def inverse(
        self, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Invert the permutation of z.

        Parameters
        ----------
        z : torch.Tensor of shape (B, D)
            Input latent data (permuted features).

        Returns
        -------
        x : torch.Tensor of shape (B, D)
            Unpermuted data (original feature ordering).
        log_det : torch.Tensor of shape (B,)
            For a permutation, it is also 0.
        """
        x = z[:, self.inv_perm]
        batch_size = z.shape[0]
        log_det = z.new_zeros(batch_size)
        return x, log_det


class MaskedAffineCoupling(nn.Module):
    """
    A Flow-style affine coupling layer with a mask that splits the features
    into two subsets: the 'masked' subset is passed through unchanged, while
    the 'unmasked' subset is scaled and translated based on an MLP applied to
    the masked subset.
    """

    def __init__(
        self,
        mask: torch.Tensor,
        scale_net: nn.Module,
        translate_net: nn.Module
    ):
        """
        Parameters
        ----------
        mask : torch.Tensor of shape (D,)
            A binary mask indicating which components of the input are active.
            (Typically 0/1 values).
        scale_net : nn.Module
            An MLP (or other net) mapping masked features -> scale parameters.
        translate_net : nn.Module
            An MLP (or other net) mapping masked features -> translate parameters.
        """
        super().__init__()
        self.register_buffer("mask", mask)  # shape (D,)
        self.scale_net = scale_net
        self.translate_net = translate_net

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: data -> latent space.

        Parameters
        ----------
        x : torch.Tensor of shape (B, D)
            Data in original space.

        Returns
        -------
        z : torch.Tensor of shape (B, D)
            Transformed data in latent space.
        log_det : torch.Tensor of shape (B,)
            The log-determinant of the Jacobian for this transform.
        """
        # Split according to mask
        x_masked = x * self.mask  # shape (B, D)

        s = self.scale_net(x_masked)      # shape (B, D)
        t = self.translate_net(x_masked)  # shape (B, D)

        z = x_masked + (1 - self.mask) * (x * torch.exp(s) + t)

        # log|det J| = sum over unmasked dimensions of s
        log_det = torch.sum(s * (1 - self.mask), dim=1)

        return z, log_det

    def inverse(
        self, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inverse pass: latent space -> data space.

        Parameters
        ----------
        z : torch.Tensor of shape (B, D)
            Data in the latent space.

        Returns
        -------
        x : torch.Tensor of shape (B, D)
            Inverted data in the original space.
        log_det : torch.Tensor of shape (B,)
            The log-determinant of the Jacobian for the inverse transform
            (i.e., negative of the forward transform's log_det).
        """
        z_masked = z * self.mask

        s = self.scale_net(z_masked)
        t = self.translate_net(z_masked)

        # Inverse for the unmasked portion:
        # x_unmasked = (z_unmasked - t) * exp(-s)
        x = z_masked + (1 - self.mask) * ((z - t) * torch.exp(-s))

        # log|det J_inv| = - log|det J_forward|
        log_det = -torch.sum(s * (1 - self.mask), dim=1)

        return x, log_det



class MLP(nn.Module):
    """
    A simple multi-layer perceptron for producing scale and translation vectors
    in an affine coupling layer.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_dim: int = 512,
        final_activation: nn.Module = None,
    ):
        """
        Parameters
        ----------
        in_features : int
            Dimension of the input features.
        out_features : int
            Dimension of the output features (e.g. same as in_features).
        hidden_dim : int
            Number of hidden units in each layer.
        final_activation : nn.Module
            Optional final activation (e.g., nn.Tanh()) to bound the scale.
        """
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hidden_dim, out_features),
        )

        # Initialize the last layer near zero
        nn.init.uniform_(self.net[-1].weight, a=-0.001, b=0.001)
        nn.init.zeros_(self.net[-1].bias)

        self.final_activation = final_activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor of shape (B, in_features)

        Returns
        -------
        out : torch.Tensor of shape (B, out_features)
        """
        out = self.net(x)
        if self.final_activation is not None:
            out = self.final_activation(out)
        return out


class Flow(nn.Module):
    """
    A normalizing flow that applies a sequence of invertible
    transformations (coupling layers + permutations). The 'forward()' method
    maps data to latent space and yields log-determinants. The 'inverse()' method
    maps latent samples back to data space.
    """

    def __init__(
        self,
        transforms: List[nn.Module],
        base_distribution: torch.distributions.Distribution,
    ):
        """
        Parameters
        ----------
        transforms : list of nn.Module
            A list of invertible transforms (coupling layers, permutations, etc.).
        base_distribution : torch.distributions.Distribution
            The base distribution (e.g., standard Normal) in the latent space.
        """
        super().__init__()
        self.transforms = nn.ModuleList(transforms)
        self.base_dist = base_distribution

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: data -> latent. We accumulate log-determinants for each transform.

        Parameters
        ----------
        x : torch.Tensor of shape (B, D)
            Data in the input (original) space.

        Returns
        -------
        z : torch.Tensor of shape (B, D)
            Data in latent space.
        log_det_total : torch.Tensor of shape (B,)
            Sum of log-determinants for all transforms.
        """
        log_det_total = 0.0
        for transform in self.transforms:
            x, log_det = transform(x)
            log_det_total += log_det
        return x, log_det_total

    def inverse(
        self, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inverse pass: latent -> data. We apply the inverse of each transform
        in reverse order.

        Parameters
        ----------
        z : torch.Tensor of shape (B, D)
            Latent representation.

        Returns
        -------
        x : torch.Tensor of shape (B, D)
            Reconstructed data in the original space.
        log_det_total : torch.Tensor of shape (B,)
            Sum of log-determinants for the inverse transforms.
        """
        log_det_total = 0.0
        for transform in reversed(self.transforms):
            z, log_det = transform.inverse(z)
            log_det_total += log_det
        return z, log_det_total

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute log p(x) for the model, i.e. log probability under the flow
        and base distribution.

        Parameters
        ----------
        x : torch.Tensor of shape (B, D)

        Returns
        -------
        log_px : torch.Tensor of shape (B,)
            The log-likelihood of the data under the flow model.
        """
        # data -> latent
        z, log_det = self.forward(x)
        # base log prob
        log_pz = self.base_dist.log_prob(z)
        # total log prob
        log_px = log_pz + log_det
        return log_px

    def sample(self, num_samples: int) -> torch.Tensor:
        """
        Generate samples from the flow model by sampling from the base and
        applying the inverse transform.

        Parameters
        ----------
        num_samples : int
            Number of samples to draw.

        Returns
        -------
        x_samples : torch.Tensor of shape (num_samples, D)
            Samples in data space.
        """
        # Sample from base distribution
        z = self.base_dist.sample((num_samples,))
        # latent -> data
        x, _ = self.inverse(z)
        return x

    def loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the negative log-likelihood (NLL) loss = -E[log p(x)].

        Parameters
        ----------
        x : torch.Tensor of shape (B, D)

        Returns
        -------
        nll : torch.Tensor
            The mean NLL over the batch.
        """
        return -torch.mean(self.log_prob(x))



def build_model(
    height: int, width: int,
    hidden_dim: int = 512,
    num_coupling_layers: int = 4,
    seed: int = 42,
    base_dist = None,
) -> Flow:
    """
    Builder for flow on flattened images of shape (height*width,).

    Parameters
    ----------
    height : int
        Image height (e.g. 28 for MNIST).
    width : int
        Image width (e.g. 28 for MNIST).
    hidden_dim : int
        Hidden dimension for MLPs in coupling layers.
    num_coupling_layers : int
        Number of coupling transformations to stack.
    seed : int
        Random seed for permutations.

    Returns
    -------
    model : Flow
        The Flow model.
    """

    D = height * width  # total features
    transforms = []

    for i in range(num_coupling_layers):
        # Create a checkerboard mask
        # Typically for 2D data, (x + y) % 2 for pixel (x,y),
        # but we flatten, so let's reconstruct 2D indices:
        mask_vals = []
        for r in range(height):
            for c in range(width):
                if (r + c) % 2 == 0:
                    mask_vals.append(1)
                else:
                    mask_vals.append(0)
        mask = torch.tensor(mask_vals, dtype=torch.float32)
        # Flip the mask on even layers
        if i % 2 == 1:
            mask = 1.0 - mask

        scale_net = MLP(D, D, hidden_dim=hidden_dim, final_activation=nn.Tanh())
        translate_net = MLP(D, D, hidden_dim=hidden_dim, final_activation=None)

        coupling = MaskedAffineCoupling(
            mask=mask,
            scale_net=scale_net,
            translate_net=translate_net
        )

        transforms.append(coupling)
        # Add a permutation layer after each coupling
        perm = InvertiblePermutation(num_features=D, seed=seed + i)
        transforms.append(perm)

    # Base distribution
    base_dist = base_dist

    flow = Flow(transforms=transforms, base_distribution=base_dist)
    return flow
