import torch
import torch.distributions as td
import torch.nn as nn
from torchvision import datasets, transforms
from tqdm import tqdm
import hydra
from priors import MoGPrior
from helpers import GAUSSIAN, MOG, VAMP


with hydra.initialize(config_path="../../../configs", version_base="1.3"):
    cfg = hydra.compose(config_name="config.yaml")


class MaskedCouplingLayer(nn.Module):
    """
    An affine coupling layer for a normalizing flow.
    """

    def __init__(self, scale_net, translation_net, mask):
        """
        Define a coupling layer.

        Parameters:
        scale_net: [torch.nn.Module]
            The scaling network that takes as input a tensor of dimension `(batch_size, feature_dim)` and outputs a tensor of dimension `(batch_size, feature_dim)`.
        translation_net: [torch.nn.Module]
            The translation network that takes as input a tensor of dimension `(batch_size, feature_dim)` and outputs a tensor of dimension `(batch_size, feature_dim)`.
        mask: [torch.Tensor]
            A binary mask of dimension `(feature_dim,)` that determines which features (where the mask is zero) are transformed by the scaling and translation networks.
        """
        super(MaskedCouplingLayer, self).__init__()
        self.scale_net = scale_net
        self.translation_net = translation_net
        self.mask = nn.Parameter(mask, requires_grad=False)

    def forward(self, z):
        """
        Transform a batch of data through the coupling layer (from the base to data).

        Parameters:
        x: [torch.Tensor]
            The input to the transformation of dimension `(batch_size, feature_dim)`
        Returns:
        z: [torch.Tensor]
            The output of the transformation of dimension `(batch_size, feature_dim)`
        sum_log_det_J: [torch.Tensor]
            The sum of the log determinants of the Jacobian matrices of the forward transformations of dimension `(batch_size, feature_dim)`.
        """
        b_z = self.mask * z
        s = self.scale_net(b_z)
        # Clamp scaling for numerical stability
        # s = torch.clamp(s, min=-2.0, max=2.0)
        t = self.translation_net(b_z)
        # Use safer exponential
        exp_s = torch.exp(s)
        z_prime = b_z + (1 - self.mask) * (z * exp_s + t)
        log_det_J = torch.sum(s * (1 - self.mask), dim=1)
        return z_prime, log_det_J

    def inverse(self, x):
        """
        Transform a batch of data through the coupling layer (from data to the base).

        Parameters:
        x: [torch.Tensor]
            The input to the inverse transformation of dimension `(batch_size, feature_dim)`
        Returns:
        z: [torch.Tensor]
            The output of the inverse transformation of dimension `(batch_size, feature_dim)`
        sum_log_det_J: [torch.Tensor]
            The sum of the log determinants of the Jacobian matrices of the inverse transformations.
        """
        z_prime = x
        b_z_prime = self.mask * z_prime
        s = self.scale_net(b_z_prime)
        # Clamp scaling for numerical stability
        # s = torch.clamp(s, min=-2.0, max=2.0)
        t = self.translation_net(b_z_prime)
        # Use safer exponential and protect division
        z = b_z_prime + (1 - self.mask) * ((z_prime - t) * torch.exp(-s))
        sum_log_det_J = -torch.sum(s * (1 - self.mask), dim=1)
        return z, sum_log_det_J


class Flow(nn.Module):
    def __init__(self, base, transformations):
        """
        Define a normalizing flow model.

        Parameters:
        base: [torch.distributions.Distribution]
            The base distribution.
        transformations: [list of torch.nn.Module]
            A list of transformations to apply to the base distribution.
        """
        super(Flow, self).__init__()
        self.base = base
        self.transformations = nn.ModuleList(transformations)

    def forward(self, x):
        """
        Transform a batch of data through the flow (from the base to data).

        Parameters:
        x: [torch.Tensor]
            The input to the transformation of dimension `(batch_size, feature_dim)`
        Returns:
        z: [torch.Tensor]
            The output of the transformation of dimension `(batch_size, feature_dim)`
        sum_log_det_J: [torch.Tensor]
            The sum of the log determinants of the Jacobian matrices of the forward transformations.
        """
        sum_log_det_J = 0
        for T in self.transformations:
            x, log_det_J = T(x)
            sum_log_det_J += log_det_J
            z = x
        return z, sum_log_det_J

    def inverse(self, x):
        """
        Transform a batch of data through the flow (from data to the base).

        Parameters:
        x: [torch.Tensor]
            The input to the inverse transformation of dimension `(batch_size, feature_dim)`
        Returns:
        z: [torch.Tensor]
            The output of the inverse transformation of dimension `(batch_size, feature_dim)`
        sum_log_det_J: [torch.Tensor]
            The sum of the log determinants of the Jacobian matrices of the inverse transformations.
        """
        sum_log_det_J = 0
        for T in reversed(self.transformations):
            x, log_det_J = T.inverse(x)
            sum_log_det_J += log_det_J
            z = x
        return z, sum_log_det_J

    def log_prob(self, x):
        """
        Compute the log probability of a batch of data under the flow.

        Parameters:
        x: [torch.Tensor]
            The data of dimension `(batch_size, feature_dim)`
        Returns:
        log_prob: [torch.Tensor]
            The log probability of the data under the flow.
        """
        z, log_det_J = self.inverse(x)
        base_log_prob = self.base().log_prob(z)
        return base_log_prob + log_det_J

    def sample(self, sample_shape=(1,)):
        """
        Sample from the flow.

        Parameters:
        n_samples: [int]
            Number of samples to generate.
        Returns:
        z: [torch.Tensor]
            The samples of dimension `(n_samples, feature_dim)`
        """
        z = self.base().sample(sample_shape)
        return self.forward(z)[0]

    def loss(self, x):
        """
        Compute the negative mean log likelihood for the given data bath.

        Parameters:
        x: [torch.Tensor]
            A tensor of dimension `(batch_size, feature_dim)`
        Returns:
        loss: [torch.Tensor]
            The negative mean log likelihood for the given data batch.
        """
        return -torch.mean(self.log_prob(x))


def build_transformations(D, num_hidden, num_transformations):
    transformations = []
    for i in range(num_transformations):
        mask = torch.randint(0, 2, (D,))

        # mask = torch.Tensor(
        #    [1 if (i + j) % 2 == 0 else 0 for i in range(28) for j in range(28)]
        # )
        scale_net = nn.Sequential(
            nn.Linear(D, num_hidden),
            nn.LeakyReLU(negative_slope=0.01),  # Smoother gradients
            nn.Linear(num_hidden, num_hidden),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(num_hidden, D),
            nn.Tanh(),  ## Added tanh for stability
        )
        translation_net = nn.Sequential(
            nn.Linear(D, num_hidden),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(num_hidden, num_hidden),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(num_hidden, D),
        )
        transformations.append(MaskedCouplingLayer(scale_net, translation_net, mask))
    return transformations
