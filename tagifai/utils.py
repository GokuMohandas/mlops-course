# utils.py
# Utility functions.

import json
import random
from pathlib import Path
from typing import Dict
from urllib.request import urlopen

import numpy as np
import torch


def load_json_from_url(url: str) -> Dict:
    """Load JSON data from a URL.

    Args:
        url (str): URL of the data source.

    Returns:
        A dictionary with the loaded JSON data.
    """
    data = json.loads(urlopen(url).read())
    return data


def create_dirs(dirpath: str) -> None:
    """Creates a directory from a specified path.

    Args:
        dirpath (str): full path of the directory to create.
    """
    Path(dirpath).mkdir(exist_ok=True)


def load_dict(filepath: str) -> Dict:
    """Load a dictionary from a JSON's filepath.

    Args:
        filepath (str): JSON's filepath.

    Returns:
        A dictionary with the data loaded.
    """
    with open(filepath, "r") as fp:
        d = json.load(fp)
    return d


def save_dict(d: Dict, filepath: str) -> None:
    """Save a dictionary to a specific location.

    Warning:
        This will overwrite any existing file at `filepath`.

    Args:
        d (Dict): dictionary to save.
        filepath (str): location to save the dictionary to as a JSON file.
    """
    with open(filepath, "w") as fp:
        json.dump(d, indent=2, sort_keys=False, fp=fp)


def set_seed(seed: int = 1234) -> None:
    """Set seed for reproducability.

    Args:
        seed (int, optional): number to use as the seed. Defaults to 1234.
    """
    # Set seeds
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # multi-GPU


def set_device(cuda: bool) -> torch.device:
    """Set the device for computation.

    Args:
        cuda (bool): Determine whether to use GPU or not (if available).

    Returns:
        Device that will be use for compute.
    """
    device = torch.device(
        "cuda" if (torch.cuda.is_available() and cuda) else "cpu"
    )
    torch.set_default_tensor_type("torch.FloatTensor")
    if device.type == "cuda":
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
    return device
