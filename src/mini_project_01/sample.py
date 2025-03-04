import hydra
import torch
from models.vae import BernoulliDecoder, GaussianEncoder, encoder_net, decoder_net
from data import load_mnist_dataset
from torchvision.utils import save_image
from helpers import get_latest_model, DEVICE
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from models.unet import Unet
import numpy as np
from scipy.stats import gaussian_kde
from helpers import GAUSSIAN

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
        if cfg.models.name == "ddpm":
            samples = model.sample((64, 784))
            samples = samples / 2 + 0.5
        elif cfg.models.name == "flow":
            temperature = 1.3 # 1 is normal, higher leads to more exploration/randomness
            samples = model.sample(64, temperature)
            # samples = samples / 2 + 0.5
        else:
            samples = model.sample(n_samples=64)
        save_image(samples.view(-1, 1, 28, 28), f"samples/sample_{cfg.models.name}.png")

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
    scatter = plt.scatter(
        all_z[:, 0], all_z[:, 1], c=all_labels, cmap="tab10", alpha=0.7
    )
    plt.colorbar(scatter, label="Class Label")
    plt.xlabel("Latent Dimension 1")
    plt.ylabel("Latent Dimension 2")
    plt.title("Posterior Samples Colored by Class Label")
    plt.savefig(f"samples/posterior_{cfg.models.name}.png")
    # plt.show()

    print(f"Posterior samples plot saved to samples/posterior_{cfg.models.name}.png")


def plot_from_prior(model, M, n_samples=200):
    """
    Plot samples from the prior distribution of the model.

    Parameters:
    model: [VAE]
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
    plt.xlabel("Latent Dimension 1")
    plt.ylabel("Latent Dimension 2")
    plt.title("Prior Samples")
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
    plt.scatter(
        all_z[:, 0], all_z[:, 1], c="blue", alpha=0.5, label="Agg. Posterior q(z)"
    )
    plt.xlabel("Latent Dim 1")
    plt.ylabel("Latent Dim 2")
    plt.title(f"Prior vs. Agg. Posterior ({cfg.models.name})")
    plt.legend()
    plt.savefig(f"samples/prior_posterior_{cfg.models.name}.png")
    print(f"Plot saved to samples/prior_posterior_{cfg.models.name}.png")


def plot_prior_contour_with_posterior(model, data_loader, M, n_samples=200):
    """
    Plot a contour of the prior density with posterior samples as black dots, handling
    higher-dimensional latent spaces and mixture priors like VampPrior.

    Parameters:
    model: [VAE]
        The VAE model to sample from.
    data_loader: [torch.utils.data.DataLoader]
        The data loader to use for sampling posterior.
    n_samples: [int]
        Number of samples to generate.
    M: [int]
        Dimension of the latent space.
    """
    print("Creating prior contour with posterior samples plot...")

    # Load the latest model
    wnb = get_latest_model(cfg.models.name)
    print(f"Loading model: {wnb}")
    model.load_state_dict(torch.load(wnb, map_location=DEVICE))

    model.eval()

    # --- Prior Contour Plot ---
    # Create a 2D grid for visualization, matching the range of your scatter plot
    x = np.linspace(-10, 10, 100)  # Expanded to match the full range of your data
    y = np.linspace(-10, 10, 100)  # Expanded to match the full range of your data
    X, Y = np.meshgrid(x, y)

    # Use the prior to generate samples or approximate density
    prior = model.prior()  # Get the prior distribution from the model

    if M > 2:
        # If the latent space is higher-dimensional, use PCA to project prior samples to 2D
        with torch.no_grad():
            # Sample from the prior to estimate its density in 2D
            n_prior_samples = 1000  # Number of samples for density estimation
            z_prior_samples = prior.sample(torch.Size([n_prior_samples])).cpu().numpy()
            pca = PCA(n_components=2)
            z_prior_2d = pca.fit_transform(z_prior_samples)

        # Use KDE to estimate 2D density from prior samples
        kde = gaussian_kde(z_prior_2d.T)
        positions = np.column_stack((X.flatten(), Y.flatten()))  # Shape: [10000, 2]
        Z = kde.evaluate(positions.T).reshape(
            100, 100
        )  # Evaluate and reshape back to grid
    else:
        # For M=2, compute density directly (e.g., for GaussianPrior)
        positions = np.dstack((X, Y))
        positions_tensor = torch.tensor(positions, dtype=torch.float32)
        log_probs = prior.log_prob(
            positions_tensor
        ).exp()  # Convert log_prob to probability
        Z = log_probs.numpy()

    # --- Posterior Samples ---
    all_z = []
    with torch.no_grad():
        for i, (x, _) in enumerate(data_loader):  # Ignore labels
            if i * data_loader.batch_size >= n_samples:
                break
            x = x.to(DEVICE)
            q = model.encoder(x)  # Get posterior distribution (q(z|x))
            z = q.rsample()  # Sample from the posterior using reparameterization
            z = z.view(-1, M)  # Ensure shape is (batch_size, M)
            all_z.append(z.cpu())
    all_z = torch.cat(all_z, dim=0).numpy()

    # Project posterior samples to 2D if M > 2
    if M > 2:
        pca = PCA(n_components=2)
        all_z = pca.fit_transform(all_z)
    elif M != 2:
        raise ValueError(
            f"Latent dimension M={M} is not supported for direct 2D plotting. Use M=2 or M>2 with PCA."
        )

    # Plot
    plt.figure(figsize=(8, 8))  # Maintain figure size but adjust plot scaling
    contour = plt.contourf(
        X, Y, Z, levels=20, cmap="viridis", alpha=1.0
    )  # Contour plot of prior density with transparency
    plt.scatter(
        all_z[:, 0], all_z[:, 1], c="black", s=10, alpha=0.5
    )  # Posterior samples as black dots
    plt.colorbar(contour, label="Density")  # Add colorbar
    plt.xlabel("Latent Dimension 1")
    plt.ylabel("Latent Dimension 2")
    plt.title("Prior Density with Posterior Samples")

    # Set axes limits to match the data range, ensuring the contour fills the space
    plt.xlim(-10, 10)  # Match the grid range
    plt.ylim(-10, 10)  # Match the grid range

    print(cfg.priors.name)
    if cfg.priors.name == GAUSSIAN:
        plt.xlim(-4, 4)
        plt.ylim(-4, 4)
    # Adjust layout to minimize white space
    plt.tight_layout()

    plt.savefig(f"samples/prior_contour_posterior_{cfg.models.name}.png")
    print(f"Plot saved to samples/prior_contour_posterior_{cfg.models.name}.png")

    plt.close()  # Close


if __name__ == "__main__":
    if cfg.models.name == "ddpm":
        net = Unet()
        model = hydra.utils.instantiate(cfg.models.model, network=net, T=cfg.T)
        sample(model)
    if cfg.models.name == "flow":
        D = 784
        # Number of transformations and hidden dim from config
        num_transformations = cfg.num_transformations_flow
        num_hidden = cfg.num_hidden_flow
        base = torch.distributions.Independent(
            torch.distributions.Normal(
                loc=torch.zeros(D).to(DEVICE), scale=torch.ones(D).to(DEVICE)
            ),
            reinterpreted_batch_ndims=1,
        )

        model = hydra.utils.instantiate(
            cfg.models.model, 28, 28, num_hidden, num_transformations, 42, base
        ).to(DEVICE)

        sample(model)
    else:
        M = cfg.latent_dim

        prior = hydra.utils.instantiate(cfg.priors.prior)

        # Define VAE model
        decoder = BernoulliDecoder(decoder_net(M))
        encoder = GaussianEncoder(encoder_net(M))

        model = hydra.utils.instantiate(
            cfg.models.model, prior=prior, decoder=decoder, encoder=encoder
        ).to(DEVICE)

        # Sample from the model
        sample(model)

        # Plot samples from the posterior
        _, test_loader = load_mnist_dataset(batch_size=cfg.training.batch_size)
        plot_from_posterior(model, test_loader, M)

        # Plot samples from the prior
        plot_from_prior(model, M)

        # Plot prior and posterior samples
        plot_prior_and_posterior(model, test_loader, M)

        plot_prior_contour_with_posterior(model, test_loader, M)
