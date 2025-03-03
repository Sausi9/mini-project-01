# import torch
import torch
import torch.nn as nn
import torch.distributions as td
import hydra
from helpers import GAUSSIAN, MOG, VAMP
# Decription: This file contains the implementation of the Variational Autoencoder (VAE) model.
# The encoder and decoder distributions are defined as separate classes.
# The prior is defined as a separate in the priors.py file.
# The VAE model is defined as a class that takes the prior, decoder, and encoder as input.


with hydra.initialize(config_path="../../../configs", version_base="1.3"):
    cfg = hydra.compose(config_name="config.yaml")


class VAE(nn.Module):
    """
    Define a Variational Autoencoder (VAE) model.
    """

    def __init__(self, prior, decoder, encoder):
        """
        Parameters:
        prior: [torch.nn.Module]
           The prior distribution over the latent space.
        decoder: [torch.nn.Module]
              The decoder distribution over the data space.
        encoder: [torch.nn.Module]
                The encoder distribution over the latent space.

        The overall VAE objective is to maximize the Evidence Lower Bound (ELBO):
        ELBO = E_{q_{phi}(z|x)}[log p_{\theta}(x|z)] - KL(q_{phi}(z|x) || p(z))
        """

        super(VAE, self).__init__()
        self.prior = prior
        self.decoder = decoder
        self.encoder = encoder

        # If using VampPrior, it needs access to the encoder
        if hasattr(self.prior, "encoder"):
            self.prior.encoder = encoder

    def elbo_gaussian(self, x):
        """
        Compute the ELBO for the given batch of data.
        Parameters:
        x: [torch.Tensor]
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2, ...)`
           n_samples: [int]
           Number of samples to use for the Monte Carlo estimate of the ELBO.
        """
        q = self.encoder(x)
        z = q.rsample()
        elbo = torch.mean(
            self.decoder(z).log_prob(x) - td.kl_divergence(q, self.prior()), dim=0
        )
        return elbo

    def elbo_mog(self, x, num_samples=5):
        """
        Compute the ELBO for the given batch of data.

        Parameters:
        x: [torch.Tensor]
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2, ...)`
           n_samples: [int]
           Number of samples to use for the Monte Carlo estimate of the ELBO.
        """
        batch_size = x.size(0)
        x_reshaped = x.view(batch_size, 28, 28) if x.dim() == 2 else x
        q = self.encoder(x)
        z = q.rsample((num_samples,))
        log_qz = q.log_prob(z)
        log_pz = self.prior().log_prob(z)

        kl_mc = torch.mean(log_qz - log_pz, dim=(0, 1))

        elbo = torch.mean(self.decoder(z).log_prob(x_reshaped), dim=(0, 1)) - kl_mc

        return elbo

    def elbo_vamp(self, x, num_samples=5):
        """
        Compute the ELBO for the given batch of data.

        Parameters:
        x: [torch.Tensor]
          A tensor of dimension `(batch_size, feature_dim1, feature_dim2, ...)`
          n_samples: [int]
          Number of samples to use for the Monte Carlo estimate of the ELBO.
        """

        batch_size = x.size(0)
        x_reshaped = x.view(batch_size, 28, 28) if x.dim() == 2 else x
        q = self.encoder(x)  # batch_shape=[batch_size], event_shape=[latent_dim]
        z = q.rsample((num_samples,))  # [num_samples, batch_size, latent_dim]
        log_qz = q.log_prob(z)  # [num_samples, batch_size]

        # Compute the mixture components from pseudo-inputs
        pseudo_inputs = self.prior.pseudo_inputs  # [K, 784]
        q_pseudo = self.encoder(
            pseudo_inputs
        )  # batch_shape=[K], event_shape=[latent_dim]
        means = q_pseudo.base_dist.loc  # [K, latent_dim]
        scales = q_pseudo.base_dist.scale  # [K, latent_dim]

        # Define the mixture distribution
        mix = td.Categorical(
            torch.ones(self.prior.K, device=means.device)
        )  # Equal weights, [K]
        comp = td.Independent(
            td.Normal(means, scales), 1
        )  # [K] batch, [latent_dim] event
        prior_mixture = td.MixtureSameFamily(
            mix, comp
        )  # batch_shape=[], event_shape=[latent_dim]

        # Compute log_pz using log_prob
        log_pz = prior_mixture.log_prob(z)  # [num_samples, batch_size]

        # Compute ELBO
        kl_mc = torch.mean(log_qz - log_pz, dim=(0, 1))  # scalar
        elbo = (
            torch.mean(self.decoder(z).log_prob(x_reshaped), dim=(0, 1)) - kl_mc
        )  # scalar

        return elbo

    def sample(self, n_samples=1):
        """
        Sample from the model.

        Parameters:
        n_samples: [int]
           Number of samples to generate.
        """
        z = self.prior().sample(torch.Size([n_samples]))
        # return self.decoder(z).sample()
        # Switch in the above line if you want to sample from the decoder distribution
        return self.decoder(z).mean

    def forward(self, x):
        """
        Compute the negative ELBO for the given batch of data.

        Parameters:
        x: [torch.Tensor]
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2)`
        """
        # If using VampPrior, sync encoder parameters before computing ELBO
        if hasattr(self.prior, "encoder"):
            with torch.no_grad():
                for p1, p2 in zip(
                    self.prior.encoder.parameters(), self.encoder.parameters()
                ):
                    p1.copy_(p2)

        prior = cfg.priors.name
        if prior == GAUSSIAN:
            elbo = self.elbo_gaussian
        if prior == MOG:
            elbo = self.elbo_mog
        if prior == VAMP:
            elbo = self.elbo_vamp
        return -elbo(x)


class GaussianEncoder(nn.Module):
    def __init__(self, encoder_net):
        """
        Define a Gaussian encoder distribution based on a given encoder network.

        q_{phi}(z|x) = N(z|mu(x), sigma(x))
        Where phi are the parameters of the encoder network i.e. weights and biases.

        Parameters:
        encoder_net: [torch.nn.Module]
           The encoder network that takes as a tensor of dim `(batch_size,
           feature_dim1, feature_dim2)` and output a tensor of dimension
           `(batch_size, 2M)`, where M is the dimension of the latent space.
        """
        super(GaussianEncoder, self).__init__()
        self.encoder_net = encoder_net

    def forward(self, x):
        """
        Given a batch of data, return a Gaussian distribution over the latent space.

        Parameters:
        x: [torch.Tensor]
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2)`
        """
        mean, std = torch.chunk(self.encoder_net(x), 2, dim=-1)
        return td.Independent(td.Normal(loc=mean, scale=torch.exp(std)), 1)


# We consider a binarized mnist, where pixels over 0.5 are considered as 1 and below as 0.
# Thus, we use a Bernoulli distribution for the decoder.
class BernoulliDecoder(nn.Module):
    def __init__(self, decoder_net):
        """
        Define a Bernoulli decoder distribution based on a given decoder network.

        p_{\theta}(x|z) = Bernoulli(x|f(z))
        Where theta are the parameters of the decoder network i.e. weights and biases

        Parameters:
        encoder_net: [torch.nn.Module]
           The decoder network that takes as a tensor of dim `(batch_size, M) as
           input, where M is the dimension of the latent space, and outputs a
           tensor of dimension (batch_size, feature_dim1, feature_dim2).
        """
        super(BernoulliDecoder, self).__init__()
        self.decoder_net = decoder_net
        self.std = nn.Parameter(torch.ones(28, 28) * 0.5, requires_grad=True)

    def forward(self, z):
        """
        Given a batch of latent variables, return a Bernoulli distribution over the data space.

        Parameters:
        z: [torch.Tensor]
           A tensor of dimension `(batch_size, M)`, where M is the dimension of the latent space.
        """
        logits = self.decoder_net(z)
        return td.Independent(td.Bernoulli(logits=logits), 2)


# Define the encoder and decoder networks for the encoder and decoder distributions.
def encoder_net(M):
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, M * 2),
    )


def decoder_net(M):
    return nn.Sequential(
        nn.Linear(M, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 784),
        nn.Unflatten(-1, (28, 28)),
    )
