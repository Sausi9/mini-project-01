import torch
import torch.nn as nn
import torch.distributions as td
import torch.utils.data
from .vae import encoder_net, GaussianEncoder


class GaussianPrior(nn.Module):
    def __init__(self, M):
        """
        Define a Gaussian prior distribution with zero mean and unit variance.

                Parameters:
        M: [int]
           Dimension of the latent space.
        """
        super(GaussianPrior, self).__init__()
        self.M = M
        self.mean = nn.Parameter(torch.zeros(self.M), requires_grad=False)
        self.std = nn.Parameter(torch.ones(self.M), requires_grad=False)

    def forward(self):
        """
        Return the prior distribution.

        Returns:
        prior: [torch.distributions.Distribution]
        """
        return td.Independent(td.Normal(loc=self.mean, scale=self.std), 1)


class MoGPrior(nn.Module):
    def __init__(self, M, K):
        """
        Define a Mixture of Gaussians prior with K components.

        Parameters:
        M: [int]
           Dimension of the latent space.
        K: [int]
           Number of Gaussian components in the mixture.
        """
        super(MoGPrior, self).__init__()
        self.M = M
        self.K = K

        # Mixture weights (logits, unnormalized)
        self.logits = nn.Parameter(torch.randn(K), requires_grad=True)

        # Gaussian means and standard deviations
        self.means = nn.Parameter(torch.randn(K, M) * 0.01)  # Reduced initial spread
        self.log_stds = nn.Parameter(
            torch.ones(K, M) * -2.0
        )  # Start with smaller variance (exp(-2) ≈ 0.135)

    def forward(self):
        """
        Return the Mixture of Gaussians prior.

        Returns:
        prior: [torch.distributions.Distribution]
        """
        # Define the categorical distribution for selecting components
        mixture_dist = td.Categorical(logits=self.logits)

        # Define the K Gaussian distributions
        component_dist = td.Independent(
            td.Normal(self.means, torch.exp(self.log_stds)), 1
        )

        # Create the MixtureSameFamily distribution
        return td.MixtureSameFamily(mixture_dist, component_dist)


# TODO: Implement VampPrior class
class VampPrior(nn.Module):
    def __init__(self, M, K, input_dim):
        super(VampPrior, self).__init__()
        self.M = M
        self.K = K

        # Create our own encoder network
        self.encoder = GaussianEncoder(encoder_net(M))

        self.pseudo_inputs = nn.Parameter(torch.randn(K, input_dim), requires_grad=True)
        self.logits = nn.Parameter(torch.zeros(K), requires_grad=True)

    def forward(self):
        """
        Return the Vamp prior.

        Returns:
        prior: [torch.distributions.Distribution]
        """

        q_pseudo = self.encoder(self.pseudo_inputs)

        # Get the base distribution (Normal) from the Independent distribution
        base_dist = q_pseudo.base_dist
        means = base_dist.loc
        stds = base_dist.scale

        mixture_dist = td.Categorical(logits=self.logits)
        component_dist = td.Independent(td.Normal(means, stds), 1)

        return td.MixtureSameFamily(mixture_dist, component_dist)
