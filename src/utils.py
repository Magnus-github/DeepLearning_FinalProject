"""Utility functions to handle object detection."""
import torch
import glob
import os


def save_model(model: torch.nn.Module, path: str) -> None:
    """Save model to disk.

    Args:
        model: The model to save.
        path: The path to save the model to.
    """
    torch.save(model.state_dict(), path)


def load_model(model: torch.nn.Module, path: str, device: str) -> torch.nn.Module:
    """Load model weights from disk.

    Args:
        model: The model to load the weights into.
        path: The path from which to load the model weights.
        device: The device the model weights should be on.

    Returns:
        The loaded model (note that this is the same object as the passed model).
    """
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    return model


def create_search_run():
    """
        Function to save the Grid Search results.
    """
    num_search_dirs = len(glob.glob('./outputs/search_*'))
    search_dirs = f"./outputs/search_{num_search_dirs+1}"
    os.makedirs(search_dirs)
    return search_dirs

