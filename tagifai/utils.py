# tagifai/utils.py
# Utility functions.

import json
import numbers
import random
from typing import Dict, List
from urllib.request import urlopen

import mlflow
import numpy as np
import pandas as pd
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


def load_dict(filepath: str) -> Dict:
    """Load a dictionary from a JSON's filepath.

    Args:
        filepath (str): JSON's filepath.

    Returns:
        A dictionary with the data loaded.
    """
    with open(filepath) as fp:
        d = json.load(fp)
    return d


def save_dict(d: Dict, filepath: str, cls=None, sortkeys: bool = False) -> None:
    """Save a dictionary to a specific location.

    Warning:
        This will overwrite any existing file at `filepath`.

    Args:
        d (Dict): dictionary to save.
        filepath (str): location to save the dictionary to as a JSON file.
        cls (optional): encoder to use on dict data. Defaults to None.
        sortkeys (bool, optional): sort keys in dict alphabetically. Defaults to False.
    """
    with open(filepath, "w") as fp:
        json.dump(d, indent=2, fp=fp, cls=cls, sort_keys=sortkeys)


def list_to_dict(list_of_dicts: List, key: str) -> Dict:
    """Convert a list of `dict_a` to a `dict_b` where
    the `key` in `dict_b` is an item in each `dict_a`.

    Args:
        list_of_dicts (List): list of items to convert to dict.
        key (str): Name of the item in `dict_a` to use as primary key for `dict_b`.

    Returns:
        A dictionary with items from the list organized by key.
    """
    d_b = {}
    for d_a in list_of_dicts:
        d_b_key = d_a.pop(key)
        d_b[d_b_key] = d_a
    return d_b


def set_seed(seed: int = 1234) -> None:
    """Set seed for reproducibility.

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
    device = torch.device("cuda" if (torch.cuda.is_available() and cuda) else "cpu")
    torch.set_default_tensor_type("torch.FloatTensor")
    if device.type == "cuda":  # pragma: no cover, simple tensor type setting
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
    return device


def dict_diff(d_a: Dict, d_b: Dict, d_a_name="a", d_b_name="b") -> Dict:
    """Differences between two dictionaries with numerical values.

    Args:
        d_a (Dict): Dictionary with data.
        d_b (Dict): Dictionary to compare to.
        d_a_name (str): Name of dict a.
        d_b_name (str): Name of dict b.

    Returns:
        Dict: Differences between keys with numerical values.
    """
    # Recursively flatten
    d_a = pd.json_normalize(d_a, sep=".").to_dict(orient="records")[0]
    d_b = pd.json_normalize(d_b, sep=".").to_dict(orient="records")[0]
    if d_a.keys() != d_b.keys():
        raise Exception("Cannot compare these dictionaries because they have different keys.")

    # Compare
    diff = {}
    for key in d_a:
        if isinstance(d_a[key], numbers.Number) and isinstance(d_b[key], numbers.Number):
            diff[key] = {d_a_name: d_a[key], d_b_name: d_b[key], "diff": d_a[key] - d_b[key]}

    return diff


def delete_experiment(experiment_name: str):
    """Delete an experiment with name `experiment_name`.

    Args:
        experiment_name (str): Name of the experiment.
    """
    client = mlflow.tracking.MlflowClient()
    experiment_id = client.get_experiment_by_name(experiment_name).experiment_id
    client.delete_experiment(experiment_id=experiment_id)
