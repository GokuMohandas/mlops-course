# main.py
# Training, optimization, etc.

import itertools
import json
import tempfile
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


def run(args: Namespace, trial: optuna.trial._trial.Trial = None) -> Dict:
    """Operations for training.

    1. Set seed
    2. Set device
    3. Load data
    4. Clean data
    5. Preprocess data
    6. Encode labels
    7. Split data
    8. Tokenize inputs
    9. Create dataloaders
    10. Initialize model
    11. Train model
    12. Evaluate model

    Args:
        args (Namespace): Input arguments for operations.
        trial (optuna.trial._trial.Trail, optional): Optuna optimization trial. Defaults to None.

    Returns:
        Artifacts to save and load for later.
    """
    # 1. Set seed
    utils.set_seed(seed=args.seed)

    # 2. Set device
    device = utils.set_device(cuda=args.cuda)

    # 3. Load data
    data_version = "0.0.1"  # TODO: Add data version from DVC
    projects_fp = Path(config.DATA_DIR, "projects.json")
    tags_fp = Path(config.DATA_DIR, "tags.json")
    projects = utils.load_dict(filepath=projects_fp)
    tags_dict = utils.list_to_dict(utils.load_dict(filepath=tags_fp), key="tag")
    df = pd.DataFrame(projects)
    if args.shuffle:
        df = df.sample(frac=1).reset_index(drop=True)
    df = df[: args.num_samples]  # None = all samples

    # 4. Clean data
    df, tags_above_frequency = data.clean(
        df=df,
        include=list(tags_dict.keys()),
        exclude=config.EXCLUDE,
        min_tag_freq=args.min_tag_freq,
    )

    # 5. Preprocess data
    df.text = df.text.apply(data.preprocess, lower=args.lower, stem=args.stem)

    # 6. Encode labels
    labels = df.tags
    label_encoder = data.MultiLabelLabelEncoder()
    label_encoder.fit(labels)
    y = label_encoder.encode(labels)

    # Class weights
    all_tags = list(itertools.chain.from_iterable(labels.values))
    counts = np.bincount([label_encoder.class_to_index[class_] for class_ in all_tags])
    class_weights = {i: 1.0 / count for i, count in enumerate(counts)}

    # 7. Split data
    utils.set_seed(seed=args.seed)  # needed for skmultilearn
    X = df.text.to_numpy()
    X_train, X_, y_train, y_ = data.iterative_train_test_split(X=X, y=y, train_size=args.train_size)
    X_val, X_test, y_val, y_test = data.iterative_train_test_split(X=X_, y=y_, train_size=0.5)
    test_df = pd.DataFrame({"text": X_test, "tags": label_encoder.decode(y_test)})

    # 8. Tokenize inputs
    tokenizer = data.Tokenizer(char_level=args.char_level)
    tokenizer.fit_on_texts(texts=X_train)
    X_train = np.array(tokenizer.texts_to_sequences(X_train), dtype=object)
    X_val = np.array(tokenizer.texts_to_sequences(X_val), dtype=object)
    X_test = np.array(tokenizer.texts_to_sequences(X_test), dtype=object)

    # 9. Create dataloaders
    train_dataset = data.CNNTextDataset(X=X_train, y=y_train, max_filter_size=args.max_filter_size)
    val_dataset = data.CNNTextDataset(X=X_val, y=y_val, max_filter_size=args.max_filter_size)
    test_dataset = data.CNNTextDataset(X=X_test, y=y_test, max_filter_size=args.max_filter_size)
    train_dataloader = train_dataset.create_dataloader(batch_size=args.batch_size)
    val_dataloader = val_dataset.create_dataloader(batch_size=args.batch_size)
    test_dataloader = test_dataset.create_dataloader(batch_size=args.batch_size)

    # 10. Initialize model
    model = models.initialize_model(
        args=args,
        vocab_size=len(tokenizer),
        num_classes=len(label_encoder),
        device=device,
    )

    # 11. Train model
    logger.info(f"Arguments: {json.dumps(args.__dict__, indent=2, cls=NumpyEncoder)}")
    args, model, loss = train.train(
        args=args,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        model=model,
        device=device,
        class_weights=class_weights,
        trial=trial,
    )

    # 12. Evaluate model
    artifacts = {
        "args": args,
        "data_version": data_version,
        "label_encoder": label_encoder,
        "tokenizer": tokenizer,
        "model": model,
        "loss": loss,
    }
    device = torch.device("cpu")
    performance, behavioral_report = eval.evaluate(
        artifacts=artifacts,
        dataloader=test_dataloader,
        df=test_df,
        device=device,
    )
    artifacts["performance"] = performance
    artifacts["behavioral_report"] = behavioral_report

    return artifacts


def objective(args: Namespace, trial: optuna.trial._trial.Trial) -> float:
    """Objective function for optimization trials.

    Args:
        args (Namespace): Input arguments for each trial (see `config/args.json`) for argument names.
        trial (optuna.trial._trial.Trial): Optuna optimization trial.

    Returns:
        F1 score from evaluating the trained model on the test data split.
    """
    # Paramters (to tune)
    args.embedding_dim = trial.suggest_int("embedding_dim", 128, 512)
    args.num_filters = trial.suggest_int("num_filters", 128, 512)
    args.hidden_dim = trial.suggest_int("hidden_dim", 128, 512)
    args.dropout_p = trial.suggest_uniform("dropout_p", 0.3, 0.8)
    args.lr = trial.suggest_loguniform("lr", 5e-5, 5e-4)

    # Train (can move some of these outside for efficiency)
    logger.info(f"\nTrial {trial.number}:")
    logger.info(json.dumps(trial.params, indent=2))
    artifacts = run(args=args, trial=trial)

    # Set additional attributes
    args = artifacts["args"]
    performance = artifacts["performance"]
    logger.info(json.dumps(performance["overall"], indent=2))
    trial.set_user_attr("threshold", args.threshold)
    trial.set_user_attr("precision", performance["overall"]["precision"])
    trial.set_user_attr("recall", performance["overall"]["recall"])
    trial.set_user_attr("f1", performance["overall"]["f1"])

    return performance["overall"]["f1"]


def load_artifacts(
    run_id: str,
    device: torch.device = torch.device("cpu"),
) -> Dict:
    """Load artifacts for a particular `run_id`.

    Args:
        run_id (str): ID of the run to load model artifacts from.
        device (torch.device): Device to run model on. Defaults to CPU.

    Returns:
        Artifacts needed for inference.
    """
    # Load model
    client = mlflow.tracking.MlflowClient()
    with tempfile.TemporaryDirectory() as dp:
        client.download_artifacts(run_id=run_id, path="", dst_path=dp)
        label_encoder = data.MultiLabelLabelEncoder.load(fp=Path(dp, "label_encoder.json"))
        tokenizer = data.Tokenizer.load(fp=Path(dp, "tokenizer.json"))
        model_state = torch.load(Path(dp, "model.pt"), map_location=device)
        performance = utils.load_dict(filepath=Path(dp, "performance.json"))

    # Load model
    run = mlflow.get_run(run_id=run_id)
    args = Namespace(**run.data.params)
    model = models.initialize_model(
        args=args, vocab_size=len(tokenizer), num_classes=len(label_encoder)
    )
    model.load_state_dict(model_state)

    return {
        "args": args,
        "label_encoder": label_encoder,
        "tokenizer": tokenizer,
        "model": model,
        "performance": performance,
    }
