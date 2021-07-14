# tagifai/main.py
# Training, optimization, etc.

import itertools
import json
from argparse import Namespace
from pathlib import Path
from typing import Dict

import mlflow
import numpy as np
import optuna
import pandas as pd
import torch
from numpyencoder import NumpyEncoder

from tagifai import config, data, eval, models, train, utils
from tagifai.config import logger


def load_artifacts(run_id: str, device: torch.device = torch.device("cpu")) -> Dict:
    """Load artifacts for current model.

    Args:
        run_id (str): ID of the model run to load artifacts. Defaults to run ID in config.MODEL_DIR.
        device (torch.device): Device to run model on. Defaults to CPU.

    Returns:
        Artifacts needed for inference.
    """
    # Load artifacts
    artifact_uri = mlflow.get_run(run_id=run_id).info.artifact_uri.split("file://")[-1]
    params = Namespace(**utils.load_dict(filepath=Path(artifact_uri, "params.json")))
    label_encoder = data.MultiLabelLabelEncoder.load(fp=Path(artifact_uri, "label_encoder.json"))
    tokenizer = data.Tokenizer.load(fp=Path(artifact_uri, "tokenizer.json"))
    model_state = torch.load(Path(artifact_uri, "model.pt"), map_location=device)
    performance = utils.load_dict(filepath=Path(artifact_uri, "performance.json"))

    # Initialize model
    model = models.initialize_model(
        params=params, vocab_size=len(tokenizer), num_classes=len(label_encoder)
    )
    model.load_state_dict(model_state)

    return {
        "params": params,
        "label_encoder": label_encoder,
        "tokenizer": tokenizer,
        "model": model,
        "performance": performance,
    }


def objective(params: Namespace, trial: optuna.trial._trial.Trial) -> float:
    """Objective function for optimization trials.

    Args:
        params (Namespace): Input parameters for each trial (see `config/params.json`).
        trial (optuna.trial._trial.Trial): Optuna optimization trial.

    Returns:
        F1 score from evaluating the trained model on the test data split.
    """
    # Paramters (to tune)
    params.embedding_dim = trial.suggest_int("embedding_dim", 128, 512)
    params.num_filters = trial.suggest_int("num_filters", 128, 512)
    params.hidden_dim = trial.suggest_int("hidden_dim", 128, 512)
    params.dropout_p = trial.suggest_uniform("dropout_p", 0.3, 0.8)
    params.lr = trial.suggest_loguniform("lr", 5e-5, 5e-4)

    # Train (can move some of these outside for efficiency)
    logger.info(f"\nTrial {trial.number}:")
    logger.info(json.dumps(trial.params, indent=2))
    artifacts = train_model(params=params, trial=trial)

    # Set additional attributes
    params = artifacts["params"]
    performance = artifacts["performance"]
    logger.info(json.dumps(performance["overall"], indent=2))
    trial.set_user_attr("threshold", params.threshold)
    trial.set_user_attr("precision", performance["overall"]["precision"])
    trial.set_user_attr("recall", performance["overall"]["recall"])
    trial.set_user_attr("f1", performance["overall"]["f1"])

    return performance["overall"]["f1"]


def compute_features(params: Namespace) -> None:
    """Compute features to use for training.

    Args:
        params (Namespace): Input parameters for operations.
    """
    # Set up
    utils.set_seed(seed=params.seed)

    # Load data
    projects_fp = Path(config.DATA_DIR, "projects.json")
    projects = utils.load_dict(filepath=projects_fp)
    df = pd.DataFrame(projects)

    # Compute features
    df["text"] = df.title + " " + df.description
    df.drop(columns=["title", "description"], inplace=True)
    df = df[["id", "created_on", "text", "tags"]]

    # Save
    features = df.to_dict(orient="records")
    df_dict_fp = Path(config.DATA_DIR, "features.json")
    utils.save_dict(d=features, filepath=df_dict_fp)

    return df, features


def train_model(params: Namespace, trial: optuna.trial._trial.Trial = None) -> Dict:
    """Operations for training.

    Args:
        params (Namespace): Input parameters for operations.
        trial (optuna.trial._trial.Trail, optional): Optuna optimization trial. Defaults to None.

    Returns:
        Artifacts to save and load for later.
    """
    # Set up
    utils.set_seed(seed=params.seed)
    device = utils.set_device(cuda=params.cuda)

    # Load features
    features_fp = Path(config.DATA_DIR, "features.json")
    tags_fp = Path(config.DATA_DIR, "tags.json")
    features = utils.load_dict(filepath=features_fp)
    tags_dict = utils.list_to_dict(utils.load_dict(filepath=tags_fp), key="tag")
    df = pd.DataFrame(features)
    if params.shuffle:
        df = df.sample(frac=1).reset_index(drop=True)
    df = df[: params.subset]  # None = all samples

    # Prepare data (filter, clean, etc.)
    df, tags_above_freq, tags_below_freq = data.prepare(
        df=df,
        include=list(tags_dict.keys()),
        exclude=config.EXCLUDED_TAGS,
        min_tag_freq=params.min_tag_freq,
    )
    params.num_samples = len(df)

    # Preprocess data
    df.text = df.text.apply(data.preprocess, lower=params.lower, stem=params.stem)

    # Encode labels
    labels = df.tags
    label_encoder = data.MultiLabelLabelEncoder()
    label_encoder.fit(labels)
    y = label_encoder.encode(labels)

    # Class weights
    all_tags = list(itertools.chain.from_iterable(labels.values))
    counts = np.bincount([label_encoder.class_to_index[class_] for class_ in all_tags])
    class_weights = {i: 1.0 / count for i, count in enumerate(counts)}

    # Split data
    utils.set_seed(seed=params.seed)  # needed for skmultilearn
    X = df.text.to_numpy()
    X_train, X_, y_train, y_ = data.iterative_train_test_split(
        X=X, y=y, train_size=params.train_size
    )
    X_val, X_test, y_val, y_test = data.iterative_train_test_split(X=X_, y=y_, train_size=0.5)
    test_df = pd.DataFrame({"text": X_test, "tags": label_encoder.decode(y_test)})

    # Tokenize inputs
    tokenizer = data.Tokenizer(char_level=params.char_level)
    tokenizer.fit_on_texts(texts=X_train)
    X_train = np.array(tokenizer.texts_to_sequences(X_train), dtype=object)
    X_val = np.array(tokenizer.texts_to_sequences(X_val), dtype=object)
    X_test = np.array(tokenizer.texts_to_sequences(X_test), dtype=object)

    # Create dataloaders
    train_dataset = data.CNNTextDataset(
        X=X_train, y=y_train, max_filter_size=params.max_filter_size
    )
    val_dataset = data.CNNTextDataset(X=X_val, y=y_val, max_filter_size=params.max_filter_size)
    train_dataloader = train_dataset.create_dataloader(batch_size=params.batch_size)
    val_dataloader = val_dataset.create_dataloader(batch_size=params.batch_size)

    # Initialize model
    model = models.initialize_model(
        params=params,
        vocab_size=len(tokenizer),
        num_classes=len(label_encoder),
        device=device,
    )

    # Train model
    logger.info(f"Parameters: {json.dumps(params.__dict__, indent=2, cls=NumpyEncoder)}")
    params, model, loss = train.train(
        params=params,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        model=model,
        device=device,
        class_weights=class_weights,
        trial=trial,
    )

    # Evaluate model
    artifacts = {
        "params": params,
        "label_encoder": label_encoder,
        "tokenizer": tokenizer,
        "model": model,
        "loss": loss,
    }
    device = torch.device("cpu")
    y_true, y_pred, performance = eval.evaluate(df=test_df, artifacts=artifacts)
    artifacts["performance"] = performance

    return artifacts
