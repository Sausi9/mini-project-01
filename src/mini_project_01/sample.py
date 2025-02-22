import hydra
from omegaconf import DictConfig
import os
from torch import nn
import torch
from models.priors import GaussianPrior
from models.vae import BernoulliDecoder, GaussianEncoder, encoder_net, decoder_net
from data import load_mnist_dataset
from torchvision.utils import save_image


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# Load configuration
with hydra.initialize(config_path="../../configs", version_base="1.3"):
    cfg = hydra.compose(config_name="config.yaml")

def get_latest_model(architecture: str = "vae") -> str:
    """Retrieve the latest model checkpoint for the given architecture."""

    # Models directory has subdirectories for each model
    models_dir = "models"
    model_dir = os.path.join(models_dir, architecture)

    # Get the latest checkpoint (based on the last modified time)
    checkpoints = os.listdir(model_dir)
    checkpoints = [os.path.join(model_dir, checkpoint) for checkpoint in checkpoints]
    latest_checkpoint = max(checkpoints, key=os.path.getmtime)

    return latest_checkpoint


def sample(model) -> None:    
    # Load the latest model
    wnb = get_latest_model(cfg.models.name)
    print(f"Loading model: {wnb}")
    model.load_state_dict(torch.load(wnb, map_location=DEVICE))

    # Sample from the model
    model.eval()
    with torch.no_grad():
        samples = model.sample(n_samples=64)
        save_image(samples.view(64, 1, 28, 28), f"samples/sample_{cfg.models.name}.png")


if __name__ == "__main__":
    M = cfg.latent_dim

    prior = hydra.utils.instantiate(cfg.priors.prior)

    # Define VAE model
    decoder = BernoulliDecoder(decoder_net(M))
    encoder = GaussianEncoder(encoder_net(M))

    model = hydra.utils.instantiate(cfg.models.model, prior=prior, decoder=decoder, encoder=encoder).to(DEVICE)
    
    sample(model)
