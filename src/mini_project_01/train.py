from models.priors import GaussianPrior, MoGPrior, VampPrior
from models.vae import VAE, BernoulliDecoder, GaussianEncoder, encoder_net, decoder_net
from data import load_mnist_dataset
import torch.nn as nn
import torch
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
import hydra
from datetime import datetime
import os


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


# Load configuration
with hydra.initialize(config_path="../../configs", version_base="1.3"):
    cfg = hydra.compose(config_name="config.yaml")

def train(model: VAE, optimizer: torch.optim.Optimizer, data_loader: torch.utils.data.DataLoader, epochs: int) -> None:
    """
    Train a VAE model.

    Parameters:
    model: [VAE]
       The VAE model to train.
    optimizer: [torch.optim.Optimizer]
         The optimizer to use for training.
    data_loader: [torch.utils.data.DataLoader]
            The data loader to use for training.
    epochs: [int]
        Number of epochs to train for.
    device: [torch.device]
        The device to use for training.
    """
    model.train()

    total_steps = len(data_loader)*epochs

    print("Training VAE...\n")
    print(f"Device: {DEVICE}")
    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))

    print("\nModel architecture:")
    print(model)

    progress_bar = tqdm(range(total_steps), desc="Training")

    for epoch in range(epochs):
        data_iter = iter(data_loader)
        for x in data_iter:
            x = x[0].to(DEVICE)
            optimizer.zero_grad()
            loss = model(x)
            loss.backward()
            optimizer.step()

            # Update progress bar
            progress_bar.set_postfix(loss=f"⠀{loss.item():12.4f}", epoch=f"{epoch+1}/{epochs}")
            progress_bar.update()

    progress_bar.close()

    print("\nTraining complete.")
    model_name = cfg.models.name
    model_path_and_name = f"models/{model_name}/{model_name}.pt"

    # Create the directory if it does not exist
    os.makedirs(os.path.dirname(model_path_and_name), exist_ok=True)

    # Save the model
    torch.save(model.state_dict(), model_path_and_name)
    

if __name__ == "__main__":

    # Load the MNIST dataset
    train_loader, _ = load_mnist_dataset(batch_size=cfg.training.batch_size)

    # Define prior distribution
    M = cfg.latent_dim

    prior = hydra.utils.instantiate(cfg.priors.prior)

    # Define VAE model
    decoder = BernoulliDecoder(decoder_net(M))
    encoder = GaussianEncoder(encoder_net(M))
    model = hydra.utils.instantiate(cfg.models.model, prior=prior, decoder=decoder, encoder=encoder).to(DEVICE)

    # Define optimizer
    optimizer = hydra.utils.instantiate(cfg.optimizer, params=model.parameters())

    train(model, optimizer, data_loader=train_loader, epochs=cfg.training.epochs)
