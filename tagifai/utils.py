# utils.py
# Utility functions.

import json
import random
from typing import Dict, List
from urllib.request import urlopen

import mlflow
import numpy as np
import torch
from tabulate import tabulate


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
    device = torch.device("cuda" if (torch.cuda.is_available() and cuda) else "cpu")
    torch.set_default_tensor_type("torch.FloatTensor")
    if device.type == "cuda":  # pragma: no cover, simple tensor type setting
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
    return device


def delete_experiment(experiment_name: str):
    """Delete an experiment with name `experiment_name`.

    Args:
        experiment_name (str): Name of the experiment.
    """
    client = mlflow.tracking.MlflowClient()
    experiment_id = client.get_experiment_by_name(experiment_name).experiment_id
    client.delete_experiment(experiment_id=experiment_id)


def get_sorted_runs(experiment_name: str, order_by: List, verbose: bool = True) -> List[Dict]:
    """Get sorted list of runs from Experiment `experiment_name`.

    Usage:

    ```python
    runs = get_sorted_runs(experiment_name="best", order_by=["metrics.f1 DESC"])
    ```

    Args:
        experiment_name (str): Name of the experiment to fetch runs from.
        order_by (List): List specification for how to order the runs.
        verbose (bool, optional): Toggle printing the table with sorted runs.

    Returns:
        List[Dict]: List of ordered runs with their respective info.
    """
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if not experiment:
        return []

    runs_df = mlflow.search_runs(
        experiment_ids=experiment.experiment_id,
        order_by=order_by,
    )

    # Convert DataFrame to List[Dict]
    runs = runs_df.to_dict("records")
    if verbose:
        print(
            tabulate(
                runs_df[
                    [
                        "run_id",
                        "end_time",
                        "metrics.f1",
                        "metrics.slices_f1",
                        "metrics.behavioral_score",
                    ]
                ],
                headers="keys",
                tablefmt="psql",
                showindex=False,
            )
        )

    return runs
