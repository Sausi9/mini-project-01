import torch
from data import load_mnist_dataset
import hydra
from models.vae import BernoulliDecoder, GaussianEncoder, encoder_net, decoder_net, VAE
from helpers import DEVICE, get_latest_model, GAUSSIAN, MOG, VAMP
import os
import numpy as np

# Load configuration
with hydra.initialize(config_path="../../configs", version_base="1.3"):
    cfg = hydra.compose(config_name="config.yaml")


def eval_elbo(model: VAE, data_loader: torch.utils.data.DataLoader, model_name: str = None) -> float:
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
    
    
    # Load the latest model if no model is provided
    wnb = model_name

    if model_name is None:
        wnb = get_latest_model(cfg.models.name)
    else:
        wnb = f"models/{cfg.models.name}/{model_name}"

    print(f"Loading model: {wnb}")
    model.load_state_dict(torch.load(wnb, map_location=DEVICE))

    model.eval()


    total_elbo = 0
    num_batches = 0

    with torch.no_grad():
        for x in data_loader:
            x = x[0].to(DEVICE)

            if cfg.priors.name == GAUSSIAN:
              elbo = model.elbo_gaussian(x)
            elif cfg.priors.name == MOG:
                elbo = model.elbo_mog(x)
            elif cfg.priors.name == VAMP:
                elbo = model.elbo_vamp(x)
            else:
                raise ValueError(f"Unknown prior type: {model.prior}")

            total_elbo += elbo.item() # Accumulate the ELBO
            num_batches += 1

    # Compute the average ELBO over all batches
    avg_elbo = total_elbo / num_batches

    return avg_elbo

def eval_elbo_mean_std(model: VAE, data_loader: torch.utils.data.DataLoader, prior: str) -> tuple[float, float]:
    """
    Fetch all the trained models with the given prior. Compute the ELBO of each model on the test data.
    Compute the mean and standard deviation of the ELBOs.

    Save corresponding model name, prior name, elbos and the mean and std of the elbos in a text file.
    """
    
    print(f"Fetching models with {prior} prior")
    vae_models = [f for f in os.listdir("models/vae") if prior in f]

    print(f"Models with {prior} prior: {vae_models}")

    # Save the model name and prior name to a text file
    with open(f"samples/elbo_{cfg.models.name}_{prior}.txt", "w") as f:
        f.write(f"Model name: {cfg.models.name}\n")
        f.write(f"Prior name: {prior}\n")
    
    elbos = []
    for model_name in vae_models:
        # Evaluate the ELBO of each model
        elbo = eval_elbo(model, data_loader, model_name)
        elbos.append(elbo)

        # Append pretrained model name and ELBO to the text file
        with open(f"samples/elbo_{cfg.models.name}_{prior}.txt", "a") as f:
            f.write(f"Model: {model_name}, ELBO: {elbo}\n")


    
    # Get the mean and standard deviation of the ELBOs
    elbos = np.array(elbos)

    mean_elbo = np.mean(elbos)
    std_elbo = np.std(elbos)

    # Append the mean and standard deviation to the text file
    with open(f"samples/elbo_{cfg.models.name}_{prior}.txt", "a") as f:
        f.write(f"\nMean ELBO: {mean_elbo}\n")
        f.write(f"Std ELBO: {std_elbo}\n")

    print("Elbo evaluation complete, results saved to file: ", f"samples/elbo_{cfg.models.name}_{prior}.txt")
    
    return mean_elbo, std_elbo



if __name__ == "__main__":
    M = cfg.latent_dim

    prior = hydra.utils.instantiate(cfg.priors.prior)

    # Define VAE model
    decoder = BernoulliDecoder(decoder_net(M))
    encoder = GaussianEncoder(encoder_net(M))

    model = hydra.utils.instantiate(cfg.models.model, prior=prior, decoder=decoder, encoder=encoder).to(DEVICE)

    # Load the MNIST dataset
    _, test_loader = load_mnist_dataset(batch_size=cfg.training.batch_size)

    # Evaluate the ELBO of models with specified prior
    mean_elbo, std_elbo = eval_elbo_mean_std(model, test_loader, cfg.priors.name)
