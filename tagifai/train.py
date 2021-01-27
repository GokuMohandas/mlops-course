# train.py
# Training operations.

from argparse import Namespace
import json
from typing import Dict, List, Tuple

import numpy as np
from numpyencoder import NumpyEncoder
import optuna
import torch
import torch.nn as nn
from sklearn.metrics import (
    precision_recall_curve,
    precision_recall_fscore_support,
)

from tagifai import data, models, utils
from tagifai.config import logger


class Trainer(object):
    """Object used to facilitate training."""

    def __init__(
        self,
        model,
        device=torch.device("cpu"),
        loss_fn=None,
        optimizer=None,
        scheduler=None,
        trial=None,
    ):

        # Set params
        self.model = model
        self.device = device
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.trial = trial

    def train_step(self, dataloader):
        """Train step.

        Args:
            dataloader (torch.utils.data.DataLoader): Torch dataloader to load batches from.

        """
        # Set model to train mode
        self.model.train()
        loss = 0.0

        # Iterate over train batches
        for i, batch in enumerate(dataloader):

            # Step
            batch = [item.to(self.device) for item in batch]  # Set device
            inputs, targets = batch[:-1], batch[-1]
            self.optimizer.zero_grad()  # Reset gradients
            z = self.model(inputs)  # Forward pass
            J = self.loss_fn(z, targets)  # Define loss
            J.backward()  # Backward pass
            self.optimizer.step()  # Update weights

            # Cumulative Metrics
            loss += (J.detach().item() - loss) / (i + 1)

        return loss

    def eval_step(self, dataloader):
        """Evaluation (val / test) step.

        Args:
            dataloader (torch.utils.data.DataLoader): Torch dataloader to load batches from.

        """
        # Set model to eval mode
        self.model.eval()
        loss = 0.0
        y_trues, y_probs = [], []

        # Iterate over val batches
        with torch.no_grad():
            for i, batch in enumerate(dataloader):

                # Step
                batch = [item.to(self.device) for item in batch]  # Set device
                inputs, y_true = batch[:-1], batch[-1]
                z = self.model(inputs)  # Forward pass
                J = self.loss_fn(z, y_true).item()

                # Cumulative Metrics
                loss += (J - loss) / (i + 1)

                # Store outputs
                y_prob = torch.sigmoid(z).cpu().numpy()
                y_probs.extend(y_prob)
                y_trues.extend(y_true.cpu().numpy())

        return loss, np.vstack(y_trues), np.vstack(y_probs)

    def predict_step(self, dataloader: torch.utils.data.DataLoader):
        """Prediction (inference) step.

        Note:
            Loss is not calculated for this loop.

        Args:
            dataloader (torch.utils.data.DataLoader): Torch dataloader to load batches from.

        """
        # Set model to eval mode
        self.model.eval()
        y_trues, y_probs = [], []

        # Iterate over val batches
        with torch.no_grad():
            for i, batch in enumerate(dataloader):

                # Forward pass w/ inputs
                batch = [item.to(self.device) for item in batch]  # Set device
                inputs, y_true = batch[:-1], batch[-1]
                z = self.model(inputs)

                # Store outputs
                y_prob = torch.sigmoid(z).cpu().numpy()
                y_probs.extend(y_prob)
                y_trues.extend(y_true.cpu().numpy())

        return np.vstack(y_trues), np.vstack(y_probs)

    def train(
        self,
        num_epochs: int,
        patience: int,
        train_dataloader: torch.utils.data.DataLoader,
        val_dataloader: torch.utils.data.DataLoader,
    ) -> Tuple:
        """Training loop.

        Args:
            num_epochs (int): Maximum number of epochs to train for (can stop earlier based on performance).
            patience (int): Number of acceptable epochs for continuous degrading performance.
            train_dataloader (torch.utils.data.DataLoader): Dataloader object with training data split.
            val_dataloader (torch.utils.data.DataLoader): Dataloader object with validation data split.

        Raises:
            optuna.TrialPruned: Early stopping of the optimization trial if poor performance.

        Returns:
            The best validation loss and the trained model from that point.
        """

        best_val_loss = np.inf
        best_model = None
        _patience = patience
        for epoch in range(num_epochs):
            # Steps
            train_loss = self.train_step(dataloader=train_dataloader)
            val_loss, _, _ = self.eval_step(dataloader=val_dataloader)
            self.scheduler.step(val_loss)

            # Pruning based on the intermediate value
            if self.trial:
                self.trial.report(val_loss, epoch)
                if self.trial.should_prune():
                    logger.info("Unpromising trial pruned!")
                    raise optuna.TrialPruned()

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = self.model
                _patience = patience  # reset _patience
            else:
                _patience -= 1
            if not _patience:  # 0
                logger.info("Stopping early!")
                break

            # Logging
            logger.info(
                f"Epoch: {epoch+1} | "
                f"train_loss: {train_loss:.5f}, "
                f"val_loss: {val_loss:.5f}, "
                f"lr: {self.optimizer.param_groups[0]['lr']:.2E}, "
                f"_patience: {_patience}"
            )
        return best_val_loss, best_model


def find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Determine the best threshold for maximum f1 score.

    Usage:

    ```python
    # Find best threshold
    _, y_true, y_prob = trainer.eval_step(dataloader=train_dataloader)
    args.threshold = find_best_threshold(y_true=y_true, y_prob=y_prob)
    ```

    Args:
        y_true (np.ndarray): True labels.
        y_prob (np.ndarray): Probability distribution for predicted labels.

    Returns:
        Best threshold for maximum f1 score.
    """
    precisions, recalls, thresholds = precision_recall_curve(
        y_true.ravel(), y_prob.ravel()
    )
    f1s = (2 * precisions * recalls) / (precisions + recalls)
    return thresholds[np.argmax(f1s)]


def get_performance(
    y_true: np.ndarray, y_pred: np.ndarray, classes: List
) -> Dict:
    """Per-class performance metrics.

    Args:
        y_true (np.ndarray): True class labels.
        y_pred (np.ndarray): Predicted class labels.
        classes (List): List of all unique classes.

    Returns:
        Dictionary of overall and per-class performance metrics.
    """
    # Get metrics
    performance = {"overall": {}, "class": {}}
    metrics = precision_recall_fscore_support(y_true, y_pred)

    # Overall performance
    performance["overall"]["precision"] = np.mean(metrics[0])
    performance["overall"]["recall"] = np.mean(metrics[1])
    performance["overall"]["f1"] = np.mean(metrics[2])
    performance["overall"]["num_samples"] = np.float64(np.sum(metrics[3]))

    # Per-class performance
    for i in range(len(classes)):
        performance["class"][classes[i]] = {
            "precision": metrics[0][i],
            "recall": metrics[1][i],
            "f1": metrics[2][i],
            "num_samples": np.float64(metrics[3][i]),
        }

    return performance


def initialize_model(
    args: Namespace,
    vocab_size: int,
    num_classes: int,
    device: torch.device = torch.device("cpu"),
) -> nn.Module:
    """Initialize a model using parameters (converted to appropriate data types).

    Args:
        args (Namespace): Parameters for data processing and training.
        vocab_size (int): Size of the vocabulary.
        num_classes (int): Number on unique classes.
        device (torch.device): Device to run model on. Defaults to CPU.
    """
    # Initialize model
    filter_sizes = list(range(1, int(args.max_filter_size) + 1))
    model = models.CNN(
        embedding_dim=int(args.embedding_dim),
        vocab_size=int(vocab_size),
        num_filters=int(args.num_filters),
        filter_sizes=filter_sizes,
        hidden_dim=int(args.hidden_dim),
        dropout_p=float(args.dropout_p),
        num_classes=int(num_classes),
    )
    model = model.to(device)
    return model


def train(
    args: Namespace,
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader,
    model: nn.Module,
    device: torch.device,
    class_weights: Dict,
    trial: optuna.trial._trial.Trial = None,
) -> Tuple:
    """Train a model.

    Args:
        args (Namespace): Parameters for data processing and training.
        train_dataloader (torch.utils.data.DataLoader): train data loader.
        val_dataloader (torch.utils.data.DataLoader): val data loader.
        model (nn.Module): Initialize model to train.
        device (torch.device): Device to run model on.
        class_weights (Dict): Dictionary of class weights.
        trial (optuna.trial._trial.Trail, optional): Optuna optimization trial. Defaults to None.

    Returns:
        The best trained model, loss and performance metrics.
    """
    # Define loss
    class_weights_tensor = torch.Tensor(np.array(list(class_weights.values())))
    loss_fn = nn.BCEWithLogitsLoss(weight=class_weights_tensor)

    # Define optimizer & scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.05, patience=5
    )

    # Trainer module
    trainer = Trainer(
        model=model,
        device=device,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        trial=trial,
    )

    # Train
    best_val_loss, best_model = trainer.train(
        args.num_epochs, args.patience, train_dataloader, val_dataloader
    )

    # Find best threshold
    _, y_true, y_prob = trainer.eval_step(dataloader=train_dataloader)
    args.threshold = find_best_threshold(y_true=y_true, y_prob=y_prob)

    return args, best_model, best_val_loss


def evaluate(
    dataloader: torch.utils.data.DataLoader,
    model: nn.Module,
    device: torch.device,
    threshold: float,
    classes: List,
) -> Dict:
    """Evaluate performance on data.

    Args:
        dataloader (torch.utils.data.DataLoader): Dataloader with the data your want to evaluate.
        model (nn.Module): Trained model.
        threshold (float): Precision recall threshold determined during training.
        classes (List): List of unique classes.
        device (torch.device): Device to run model on. Defaults to CPU.

    Returns:
        Performance metrics.
    """
    # Determine predictions using threshold
    trainer = Trainer(model=model, device=device)
    y_true, y_prob = trainer.predict_step(dataloader=dataloader)
    y_pred = np.array([np.where(prob >= threshold, 1, 0) for prob in y_prob])

    # Evaluate
    performance = get_performance(
        y_true=y_true, y_pred=y_pred, classes=classes
    )

    return performance


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
    df, projects_dict, tags_dict = data.load(
        shuffle=args.shuffle, num_samples=args.num_samples
    )

    # 4. Clean data
    df, tags_dict, tags_above_frequency = data.clean(
        df=df, tags_dict=tags_dict, min_tag_freq=args.min_tag_freq
    )

    # 5. Preprocess data
    df.text = df.text.apply(data.preprocess, lower=args.lower, stem=args.stem)

    # 6. Encode labels
    y, class_weights, label_encoder = data.encode_labels(labels=df.tags)

    # 7. Split data
    utils.set_seed(seed=args.seed)  # needed for skmultilearn
    X_train, X_val, X_test, y_train, y_val, y_test = data.split(
        X=df.text.to_numpy(), y=y, train_size=args.train_size
    )

    # 8. Tokenize inputs
    X_train, tokenizer = data.tokenize_text(
        X=X_train, char_level=args.char_level
    )
    X_val, _ = data.tokenize_text(
        X=X_val, char_level=args.char_level, tokenizer=tokenizer
    )
    X_test, _ = data.tokenize_text(
        X=X_test, char_level=args.char_level, tokenizer=tokenizer
    )

    # 9. Create dataloaders
    train_dataloader = data.get_dataloader(
        data=[X_train, y_train],
        max_filter_size=args.max_filter_size,
        batch_size=args.batch_size,
    )
    val_dataloader = data.get_dataloader(
        data=[X_val, y_val],
        max_filter_size=args.max_filter_size,
        batch_size=args.batch_size,
    )
    test_dataloader = data.get_dataloader(
        data=[X_test, y_test],
        max_filter_size=args.max_filter_size,
        batch_size=args.batch_size,
    )

    # 10. Initialize model
    model = initialize_model(
        args=args,
        vocab_size=len(tokenizer),
        num_classes=len(label_encoder),
        device=device,
    )

    # 11. Train model
    logger.info(
        f"Arguments: {json.dumps(args.__dict__, indent=2, cls=NumpyEncoder)}"
    )
    args, model, loss = train(
        args=args,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        model=model,
        device=device,
        class_weights=class_weights,
        trial=trial,
    )

    # 12. Evaluate model
    device = torch.device("cpu")
    performance = evaluate(
        dataloader=test_dataloader,
        model=model.to(device),
        device=device,
        threshold=args.threshold,
        classes=label_encoder.classes,
    )

    return {
        "args": args,
        "label_encoder": label_encoder,
        "tokenizer": tokenizer,
        "model": model,
        "loss": loss,
        "performance": performance,
    }


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
