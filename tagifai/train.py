# tagifai/train.py
# Training operations.

from argparse import Namespace
from typing import Dict, Tuple

import numpy as np
import optuna
import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_curve

from tagifai.config import logger


class Trainer:
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
                if self.trial.should_prune():  # pragma: no cover, optuna pruning
                    logger.info("Unpromising trial pruned!")
                    raise optuna.TrialPruned()

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = self.model
                _patience = patience  # reset _patience
            else:  # pragma: no cover, simple subtraction
                _patience -= 1
            if not _patience:  # pragma: no cover, simple break
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
    params.threshold = find_best_threshold(y_true=y_true, y_prob=y_prob)
    ```

    Args:
        y_true (np.ndarray): True labels.
        y_prob (np.ndarray): Probability distribution for predicted labels.

    Returns:
        Best threshold for maximum f1 score.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true.ravel(), y_prob.ravel())
    f1s = (2 * precisions * recalls) / (precisions + recalls)
    return thresholds[np.argmax(f1s)]


def train(
    params: Namespace,
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader,
    model: nn.Module,
    device: torch.device,
    class_weights: Dict,
    trial: optuna.trial._trial.Trial = None,
) -> Tuple:
    """Train a model.

    Args:
        params (Namespace): Parameters for data processing and training.
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
    optimizer = torch.optim.Adam(model.parameters(), lr=params.lr)
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
        params.num_epochs, params.patience, train_dataloader, val_dataloader
    )

    # Find best threshold
    _, y_true, y_prob = trainer.eval_step(dataloader=train_dataloader)
    params.threshold = find_best_threshold(y_true=y_true, y_prob=y_prob)

    return params, best_model, best_val_loss
