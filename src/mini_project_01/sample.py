import hydra
from omegaconf import DictConfig
import os
from torch import nn
import torch
from models.priors import GaussianPrior
from models.vae import VAE, BernoulliDecoder, GaussianEncoder, encoder_net, decoder_net
from data import load_mnist_dataset
from torchvision.utils import save_image
from helpers import get_latest_model, DEVICE
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# Load configuration
with hydra.initialize(config_path="../../configs", version_base="1.3"):
    cfg = hydra.compose(config_name="config.yaml")

def sample(model) -> None:
    """
    Sample from the model and save the samples to a file.

    Parameters:
    model: [VAE]
        The VAE model to sample from.
    """
    print("Sampling from the learned model...")

    # Load the latest model
    wnb = get_latest_model(cfg.models.name)
    print(f"Loading model: {wnb}")
    model.load_state_dict(torch.load(wnb, map_location=DEVICE))

    # Sample from the model
    model.eval()
    with torch.no_grad():
        samples = model.sample(n_samples=64)
        save_image(samples.view(64, 1, 28, 28), f"samples/sample_{cfg.models.name}.png")

    print(f"Samples saved to samples/sample_{cfg.models.name}.png")


def plot_from_posterior(model, data_loader, M, n_samples=200):
    """
    Plot samples from the posterior distribution of the model.

    Parameters:
    model: [VAE] 
        The VAE model to sample from.
    data_loader: [torch.utils.data.DataLoader]
        The data loader to use for sampling.
    n_samples: [int]
        Number of samples to generate.
    M: [int]
        Dimension of the latent space.
    """

    print("Creating posterior samples plot...")

    # Load the latest model
    wnb = get_latest_model(cfg.models.name)
    print(f"Loading model: {wnb}")
    model.load_state_dict(torch.load(wnb, map_location=DEVICE))

    model.eval()

    all_z = []
    all_labels = []

    with torch.no_grad():
        for i, (x, labels) in enumerate(data_loader):
            if i * data_loader.batch_size >= n_samples:
                break
            x = x.to(DEVICE)
            q = model.encoder(x)
            z = q.rsample()
            z = z.view(-1, M)
            all_z.append(z.cpu())
            all_labels.append(labels)

    all_z = torch.cat(all_z, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()

    # Perform PCA if M > 2
    if M > 2:
        pca = PCA(n_components=2)
        all_z = pca.fit_transform(all_z)

    # Plot the samples
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(all_z[:, 0], all_z[:, 1], c=all_labels, cmap='tab10', alpha=0.7)
    plt.colorbar(scatter, label='Class Label')
    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    plt.title('Posterior Samples Colored by Class Label')
    plt.savefig(f"samples/posterior_{cfg.models.name}.png")
    # plt.show()

    print(f"Posterior samples plot saved to samples/posterior_{cfg.models.name}.png")


def plot_from_prior(model, M, n_samples=200):
    """
    Plot samples from the prior distribution of the model.

    Parameters:
    model: [VAE] 
        The VAE model to sample from.
    n_samples: [int]
        Number of samples to generate.
    M: [int]
        Dimension of the latent space.
    """
    print("Creating prior samples plot...")

    model.eval()

    with torch.no_grad():
        z_prior = model.prior().sample(torch.Size([n_samples]))  # Shape: (n_samples, M)
        z_prior = z_prior.cpu().numpy()

    # Perform PCA if M > 2
    if M > 2:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        z_prior = pca.fit_transform(z_prior)

    # Plot the samples
    plt.figure(figsize=(8, 6))
    plt.scatter(z_prior[:, 0], z_prior[:, 1], c="green", alpha=0.7, label="Prior p(z)")
    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    plt.title('Prior Samples')
    plt.legend()
    plt.savefig(f"samples/prior_{cfg.models.name}.png")
    print(f"Prior samples plot saved to samples/prior_{cfg.models.name}.png")

def plot_prior_and_posterior(model, data_loader, M, n_samples=200):
    model.eval()
    wnb = get_latest_model(cfg.models.name)
    print(f"Loading model: {wnb}")
    model.load_state_dict(torch.load(wnb, map_location=DEVICE))

    # Prior
    with torch.no_grad():
        z_prior = model.prior().sample(torch.Size([n_samples])).cpu().numpy()

    # Aggregate posterior
    all_z = []
    with torch.no_grad():
        for i, (x, _) in enumerate(data_loader):  # Ignore labels
            if i * data_loader.batch_size >= n_samples:
                break
            x = x.to(DEVICE)
            q = model.encoder(x)
            z = q.mean
            all_z.append(z.cpu())
    all_z = torch.cat(all_z, dim=0).numpy()

    # PCA if M > 2
    if M > 2:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        z_prior = pca.fit_transform(z_prior)
        all_z = pca.fit_transform(all_z)

    # Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(z_prior[:, 0], z_prior[:, 1], c="green", alpha=0.5, label="Prior p(z)")
    plt.scatter(all_z[:, 0], all_z[:, 1], c="blue", alpha=0.5, label="Agg. Posterior q(z)")
    plt.xlabel('Latent Dim 1')
    plt.ylabel('Latent Dim 2')
    plt.title(f'Prior vs. Agg. Posterior ({cfg.models.name})')
    plt.legend()
    plt.savefig(f"samples/prior_posterior_{cfg.models.name}.png")
    print(f"Plot saved to samples/prior_posterior_{cfg.models.name}.png")

if __name__ == "__main__":
    M = cfg.latent_dim

    prior = hydra.utils.instantiate(cfg.priors.prior)

    # Define VAE model
    decoder = BernoulliDecoder(decoder_net(M))
    encoder = GaussianEncoder(encoder_net(M))

    model = hydra.utils.instantiate(cfg.models.model, prior=prior, decoder=decoder, encoder=encoder).to(DEVICE)
    
    # Sample from the model
    sample(model)

    # Plot samples from the posterior
    _, test_loader = load_mnist_dataset(batch_size=cfg.training.batch_size)
    plot_from_posterior(model, test_loader, M)
    

    # Plot samples from the prior
    plot_from_prior(model, M)

    # Plot prior and posterior samples
    plot_prior_and_posterior(model, test_loader, M)
