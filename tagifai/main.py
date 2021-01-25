# main.py
# Main operations.

import json
import tempfile
import warnings
from argparse import Namespace
from pathlib import Path
from typing import Dict

import mlflow
import optuna
import torch
import typer
import yaml
from numpyencoder import NumpyEncoder
from optuna.integration.mlflow import MLflowCallback

from tagifai import config, predict, train, utils
from tagifai.config import logger

# Ignore warning
warnings.filterwarnings("ignore")

# Typer CLI app
app = typer.Typer()


@app.command()
def download_data():
    """Download data from online to local drive.
    We could've just copied files from `datasets/` but
    we'll use this later on with other data sources.
    """
    # Download data
    projects_url = "https://raw.githubusercontent.com/GokuMohandas/applied-ml/main/datasets/projects.json"
    tags_url = "https://raw.githubusercontent.com/GokuMohandas/applied-ml/main/datasets/tags.json"
    projects = utils.load_json_from_url(url=projects_url)
    tags = utils.load_json_from_url(url=tags_url)

    # Save data
    projects_fp = Path(config.DATA_DIR, "projects.json")
    tags_fp = Path(config.DATA_DIR, "tags.json")
    utils.save_dict(d=projects, filepath=projects_fp)
    utils.save_dict(d=tags, filepath=tags_fp)
    logger.info("✅ Data downloaded!")


@app.command()
def train_model(args_fp: Path = Path(config.CONFIG_DIR, "args.json")) -> None:
    """Train a model using the specified parameters.

    Args:
        args_fp (Path, optional): Location of arguments to use for training. Defaults to Path(config.CONFIG_DIR, "args.json").
    """
    # Set experiment and start run
    args = Namespace(**utils.load_dict(filepath=args_fp))

    # Start run
    mlflow.set_experiment(experiment_name="best")
    with mlflow.start_run(
        run_name="cnn"
    ) as run:  # NOQA: F841 (assigned to but never used)
        # Train
        artifacts = train.run(args=args)

        # Log key metrics
        performance = artifacts["performance"]
        loss = artifacts["loss"]
        mlflow.log_metrics({"precision": performance["overall"]["precision"]})
        mlflow.log_metrics({"recall": performance["overall"]["recall"]})
        mlflow.log_metrics({"f1": performance["overall"]["f1"]})
        mlflow.log_metrics({"best_val_loss": loss})

        # Log artifacts
        args = artifacts["args"]
        model = artifacts["model"]
        label_encoder = artifacts["label_encoder"]
        tokenizer = artifacts["tokenizer"]
        with tempfile.TemporaryDirectory() as fp:
            label_encoder.save(Path(fp, "label_encoder.json"))
            tokenizer.save(Path(fp, "tokenizer.json"))
            torch.save(model.state_dict(), Path(fp, "model.pt"))
            utils.save_dict(performance, Path(fp, "performance.json"))
            mlflow.log_artifacts(fp)
        mlflow.log_params(vars(args))
    logger.info(json.dumps(performance["overall"], indent=2))


@app.command()
def predict_tags(
    text: str = "Transfer learning with BERT for self-supervised learning",
    run_id: str = "",
) -> Dict:
    """Predict tags for a give input text using a trained model. Make sure that you have a trained model first!

    Args:
        text (str, optional): Input text to predict tags for. Defaults to "Transfer learning with BERT for self-supervised learning".
        run_id (str, optional): ID of the run to load model artifacts from. Defaults to model with lowest `best_val_loss` from the `best` experiment.

    Returns:
        Predicted tags for input text.
    """
    # Get best run
    if not run_id:
        experiment_id = mlflow.get_experiment_by_name("best").experiment_id
        all_runs = mlflow.search_runs(
            experiment_ids=experiment_id,
            order_by=["metrics.best_val_loss ASC"],
        )
        run_id = all_runs.iloc[0].run_id

    # Predict
    prediction = predict.predict(texts=[text], run_id=run_id)
    logger.info(json.dumps(prediction, indent=2))

    return prediction


@app.command()
def optimize(num_trials: int = 100) -> None:
    """Optimize a subset of hyperparameters towards an objective.

    Args:
        num_trials (int, optional): Number of trials to run. Defaults to 100.
    """
    # Starting arguments (not actually used but needed for set up)
    args_fp = Path(config.CONFIG_DIR, "args.json")
    args = Namespace(**utils.load_dict(filepath=args_fp))

    # Optimize
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    study = optuna.create_study(
        study_name="optimization", direction="minimize", pruner=pruner
    )
    mlflow_callback = MLflowCallback(
        tracking_uri=mlflow.get_tracking_uri(), metric_name="best_val_loss"
    )
    study.optimize(
        lambda trial: train.objective(args, trial),
        n_trials=num_trials,
        callbacks=[mlflow_callback],
    )

    # All trials
    trials_df = study.trials_dataframe()
    trials_df = trials_df.sort_values(
        ["value"], ascending=True
    )  # sort by metric
    trials_df.to_csv(
        Path(config.EXPERIMENTS_DIR, "trials.csv"), index=False
    )  # save

    # Best trial
    logger.info(f"Best value (`best_val_loss`): {study.best_trial.value}")
    params = {**args.__dict__, **study.best_trial.params}
    params["threshold"] = study.best_trial.user_attrs["threshold"]
    with open(Path(config.CONFIG_DIR, "args.json"), "w") as fp:
        json.dump(params, fp=fp, indent=2, cls=NumpyEncoder)
    logger.info(json.dumps(params, indent=2, cls=NumpyEncoder))


@app.command()
def set_artifact_metadata():
    """Set the artifact URI for all experiments and runs.
    Used when transferring experiments from other locations (ex. Colab).
    """

    def set_artifact_location(var, fp):
        """Set variable's yaml value on file at fp."""
        with open(fp) as f:
            metadata = yaml.load(f)

        # Set new value
        experiment_id = metadata[var].split("/")[-1]
        artifact_location = Path(
            "file://", config.EXPERIMENTS_DIR, experiment_id
        )
        metadata[var] = str(artifact_location)

        with open(fp, "w") as f:
            yaml.dump(metadata, f)

    def set_artifact_uri(var, fp):
        """Set variable's yaml value on file at fp."""
        with open(fp) as f:
            metadata = yaml.load(f)

        # Set new value
        experiment_id = metadata[var].split("/")[-3]
        run_id = metadata[var].split("/")[-2]
        artifact_uri = Path(
            "file://",
            config.EXPERIMENTS_DIR,
            experiment_id,
            run_id,
            "artifacts",
        )
        metadata[var] = str(artifact_uri)

        with open(fp, "w") as f:
            yaml.dump(metadata, f)

    # Get artifact location
    experiment_meta_yamls = list(
        Path(config.EXPERIMENTS_DIR).glob("*/meta.yaml")
    )
    for meta_yaml in experiment_meta_yamls:
        set_artifact_location(var="artifact_location", fp=meta_yaml)

    # Change artifact URI
    run_meta_yamls = list(Path(config.EXPERIMENTS_DIR).glob("*/*/meta.yaml"))
    for meta_yaml in run_meta_yamls:
        set_artifact_uri(var="artifact_uri", fp=meta_yaml)


if __name__ == "__main__":
    app()
