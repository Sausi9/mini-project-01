import torch
from data import load_mnist_dataset
import hydra
from models.vae import BernoulliDecoder, GaussianEncoder, encoder_net, decoder_net, VAE
from helpers import DEVICE, get_latest_model

# Load configuration
with hydra.initialize(config_path="../../configs", version_base="1.3"):
    cfg = hydra.compose(config_name="config.yaml")


def eval_elbo(model: VAE, data_loader: torch.utils.data.DataLoader) -> float:
    """
    Evaluate the ELBO of a VAE model on a dataset.

    Parameters:
    model: [VAE]
        The VAE model to evaluate.
    data_loader: [torch.utils.data.DataLoader]
        The test data used for evaluation.

    Returns:
    avg_elbo: [float]
        The average ELBO of the model on the test data.
    """

    # Load the latest model
    wnb = get_latest_model(cfg.models.name)
    print(f"Loading model: {wnb}")
    model.load_state_dict(torch.load(wnb, map_location=DEVICE))

    model.eval()

    total_elbo = 0
    num_batches = 0

    with torch.no_grad():
        for x in data_loader:
            x = x[0].to(DEVICE)
            elbo = model.elbo(x) # Compute the ELBO for the batch
            total_elbo += elbo.item() # Accumulate the ELBO
            num_batches += 1
    
    # Compute the average ELBO over all batches
    avg_elbo = total_elbo / num_batches

    return avg_elbo




if __name__ == "__main__":
    M = cfg.latent_dim

    prior = hydra.utils.instantiate(cfg.priors.prior)

    # Define VAE model
    decoder = BernoulliDecoder(decoder_net(M))
    encoder = GaussianEncoder(encoder_net(M))

    model = hydra.utils.instantiate(cfg.models.model, prior=prior, decoder=decoder, encoder=encoder).to(DEVICE)

    # Load the MNIST dataset
    _, test_loader = load_mnist_dataset(batch_size=cfg.training.batch_size)

    elbo = eval_elbo(model, test_loader)
    print(f"ELBO: {elbo}")