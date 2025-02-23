import os
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


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