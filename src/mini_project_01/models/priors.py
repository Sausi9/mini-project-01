import torch
import torch.nn as nn
import torch.distributions as td
import torch.utils.data


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

        # Gaussian means and standard deviations (initialized around 0 and 1)
        self.means = nn.Parameter(torch.randn(K, M), requires_grad=True)
        self.stds = nn.Parameter(torch.ones(K, M), requires_grad=True)

    def forward(self):
        """
        Return the Mixture of Gaussians prior.

        Returns:
        prior: [torch.distributions.Distribution]
        """
        # Define the categorical distribution for selecting components
        mixture_dist = td.Categorical(logits=self.logits)

        # Define the K Gaussian distributions
        component_dist = td.Independent(td.Normal(self.means, self.stds), 1)

        # Create the MixtureSameFamily distribution
        return td.MixtureSameFamily(mixture_dist, component_dist)
    

# TODO: Implement VampPrior class