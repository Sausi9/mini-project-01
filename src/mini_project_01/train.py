from models.unet import Unet
from models.vae import BernoulliDecoder, GaussianEncoder, encoder_net, decoder_net
from data import load_mnist_dataset
import torch
from tqdm import tqdm
from omegaconf import OmegaConf
import hydra
import os


DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


# Load configuration
with hydra.initialize(config_path="../../configs", version_base="1.3"):
    cfg = hydra.compose(config_name="config.yaml")


def train(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    data_loader: torch.utils.data.DataLoader,
    epochs: int,
    scheduler: torch.optim.lr_scheduler._LRScheduler = None,
) -> None:
    """
    Train a model for a specified number of epochs.

    Parameters
    ----------
    model : torch.nn.Module
        The model to train. Must support `.train()` and either `.loss(x)` or
        a forward call returning a loss.
    optimizer : torch.optim.Optimizer
        The optimizer to use for gradient-based updates.
    data_loader : torch.utils.data.DataLoader
        Data loader providing the training samples.
    epochs : int
        Number of epochs to train for.
    scheduler : torch.optim.lr_scheduler._LRScheduler, optional
        Optional learning-rate scheduler. If provided, `scheduler.step()` is
        called at the end of each epoch.
    """
    model.train()

    total_steps = len(data_loader) * epochs

    print("Training model...\n")
    print(f"Device: {DEVICE}")
    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))

    print("\nModel architecture:")
    print(model)

    progress_bar = tqdm(range(total_steps), desc="Training")

    for epoch in range(epochs):
        data_iter = iter(data_loader)
        for x in data_iter:
            if isinstance(x, (list, tuple)):
                x = x[0]
            x = x.to(DEVICE)
            optimizer.zero_grad()
            if cfg.models.name == "ddpm" or cfg.models.name == "flow":
                loss = model.loss(x)
            else:
                loss = model(x)
            loss.backward()
            optimizer.step()

            # Update progress bar
            progress_bar.set_postfix(
                loss=f"⠀{loss.item():12.4f}", epoch=f"{epoch+1}/{epochs}"
            )
            progress_bar.update()
        # Step the scheduler once per epoch
        if scheduler is not None:
            scheduler.step()

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
    train_loader, _ = load_mnist_dataset(
        batch_size=cfg.training.batch_size, binarized=cfg.training.binarized, 
    )
    scheduler = None
    if cfg.models.name == "vae":
        # Define prior distribution
        M = cfg.latent_dim

        prior = hydra.utils.instantiate(cfg.priors.prior)

        # Define VAE model
        decoder = BernoulliDecoder(decoder_net(M))
        encoder = GaussianEncoder(encoder_net(M))
        model = hydra.utils.instantiate(
            cfg.models.model, prior=prior, decoder=decoder, encoder=encoder
        ).to(DEVICE)
    elif cfg.models.name == "ddpm":
        net = Unet().to(DEVICE)
        model = hydra.utils.instantiate(cfg.models.model, network=net, T=cfg.T).to(
            DEVICE
        )
    elif cfg.models.name == "flow":
        train_loader, _ = load_mnist_dataset(
            batch_size=cfg.training.batch_size, binarized=cfg.training.binarized, do_logit=True,
        )
        D = 784
        # Number of transformations and hidden dim from config
        num_transformations = cfg.num_transformations_flow
        num_hidden = cfg.num_hidden_flow
        base = torch.distributions.Independent(
            torch.distributions.Normal(loc=torch.zeros(D).to(DEVICE), scale=torch.ones(D).to(DEVICE)),
            reinterpreted_batch_ndims=1,
        )

        model = hydra.utils.instantiate(
            cfg.models.model, 28, 28, num_hidden, num_transformations, 42, base
        ).to(DEVICE)

    # Define optimizer
    optimizer = hydra.utils.instantiate(cfg.optimizer, params=model.parameters())
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1)
    train(
        model,
        optimizer,
        data_loader=train_loader,
        epochs=cfg.training.epochs,
        scheduler=scheduler,
    )
