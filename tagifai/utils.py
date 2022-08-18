import json
import random
from typing import Dict

import numpy as np


def load_dict(filepath: str) -> Dict:
    """Load a dictionary from a JSON's filepath.

    Args:
        filepath (str): location of file.

    Returns:
        Dict: loaded JSON data.
    """
    with open(filepath) as fp:
        d = json.load(fp)
    return d


def save_dict(d: Dict, filepath: str, cls=None, sortkeys: bool = False) -> None:
    """Save a dictionary to a specific location.

    Args:
        d (Dict): data to save.
        filepath (str): location of where to save the data.
        cls (optional): encoder to use on dict data. Defaults to None.
        sortkeys (bool, optional): whether to sort keys alphabetically. Defaults to False.
    """
    with open(filepath, "w") as fp:
        json.dump(d, indent=2, fp=fp, cls=cls, sort_keys=sortkeys)
        fp.write("\n")


def set_seeds(seed: int = 42) -> None:
    """Set seed for reproducibility.

    Args:
        seed (int, optional): number to be used as the seed. Defaults to 42.
    """
    # Set seeds
    np.random.seed(seed)
    random.seed(seed)
