# app/cli.py
# Command line interface (CLI) application.

import json
import numbers
import tempfile
import warnings
from argparse import Namespace
from pathlib import Path
from typing import Dict, Optional

import mlflow
import optuna
import pandas as pd
import torch
import typer
from numpyencoder import NumpyEncoder
from optuna.integration.mlflow import MLflowCallback

from tagifai import config, eval, main, predict, utils
from tagifai.config import logger

# Ignore warning
warnings.filterwarnings("ignore")

# Typer CLI app
app = typer.Typer()


@app.command()
def download_data():
    """Download data from online to local drive.

    Note:
        We could've just copied files from `datasets` but
        we'll use this later on with other data sources.
    """
    # Download data
    projects_url = (
        "https://raw.githubusercontent.com/GokuMohandas/madewithml/main/datasets/projects.json"
    )
    tags_url = "https://raw.githubusercontent.com/GokuMohandas/madewithml/main/datasets/tags.json"
    projects = utils.load_json_from_url(url=projects_url)
    tags = utils.load_json_from_url(url=tags_url)

    # Save data
    projects_fp = Path(config.DATA_DIR, "projects.json")
    tags_fp = Path(config.DATA_DIR, "tags.json")
    utils.save_dict(d=projects, filepath=projects_fp)
    utils.save_dict(d=tags, filepath=tags_fp)
    logger.info("✅ Data downloaded!")


@app.command()
def optimize(
    params_fp: Path = Path(config.CONFIG_DIR, "params.json"),
    study_name: Optional[str] = "optimization",
    num_trials: int = 100,
) -> None:
    """Optimize a subset of hyperparameters towards an objective.

    This saves the best trial's parameters into `config/params.json`.

    Args:
        params_fp (Path, optional): Location of parameters (just using num_samples,
                                  num_epochs, etc.) to use for training.
                                  Defaults to `config/params.json`.
        study_name (str, optional): Name of the study to save trial runs under. Defaults to `optimization`.
        num_trials (int, optional): Number of trials to run. Defaults to 100.
    """
    # Starting parameters (not actually used but needed for set up)
    params = Namespace(**utils.load_dict(filepath=params_fp))

    # Optimize
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    study = optuna.create_study(study_name=study_name, direction="maximize", pruner=pruner)
    mlflow_callback = MLflowCallback(tracking_uri=mlflow.get_tracking_uri(), metric_name="f1")
    study.optimize(
        lambda trial: main.objective(params, trial),
        n_trials=num_trials,
        callbacks=[mlflow_callback],
    )

    # All trials
    trials_df = study.trials_dataframe()
    trials_df = trials_df.sort_values(["value"], ascending=False)

    # Best trial
    logger.info(f"Best value (f1): {study.best_trial.value}")
    params = {**params.__dict__, **study.best_trial.params}
    params["threshold"] = study.best_trial.user_attrs["threshold"]
    with open(params_fp, "w") as fp:
        json.dump(params, fp=fp, indent=2, cls=NumpyEncoder)
    logger.info(json.dumps(params, indent=2, cls=NumpyEncoder))


@app.command()
def train_model(
    params_fp: Path = Path(config.CONFIG_DIR, "params.json"),
    model_dir: Optional[Path] = Path(config.MODEL_DIR),
    experiment_name: Optional[str] = "best",
    run_name: Optional[str] = "model",
) -> None:
    """Train a model using the specified parameters.

    Args:
        params_fp (Path, optional): Parameters to use for training. Defaults to `config/params.json`.
        model_dir (Path): location of model artifacts. Defaults to config.MODEL_DIR.
        experiment_name (str, optional): Name of the experiment to save the run to. Defaults to `best`.
        run_name (str, optional): Name of the run. Defaults to `model`.
    """
    # Set experiment and start run
    params = Namespace(**utils.load_dict(filepath=params_fp))

    # Start run
    mlflow.set_experiment(experiment_name=experiment_name)
    with mlflow.start_run(run_name=run_name):
        # Train
        artifacts = main.run(params=params)

        # Set tags
        tags = {}
        mlflow.set_tags(tags)

        # Log metrics
        performance = artifacts["performance"]
        logger.info(json.dumps(performance["overall"], indent=2))
        metrics = {
            "precision": performance["overall"]["precision"],
            "recall": performance["overall"]["recall"],
            "f1": performance["overall"]["f1"],
            "best_val_loss": artifacts["loss"],
            "behavioral_score": performance["behavioral"]["score"],
            "slices_f1": performance["slices"]["overall"]["f1"],
        }
        mlflow.log_metrics(metrics)

        # Log artifacts
        with tempfile.TemporaryDirectory() as dp:
            artifacts["label_encoder"].save(Path(dp, "label_encoder.json"))
            artifacts["tokenizer"].save(Path(dp, "tokenizer.json"))
            torch.save(artifacts["model"].state_dict(), Path(dp, "model.pt"))
            utils.save_dict(performance, Path(dp, "performance.json"))
            mlflow.log_artifacts(dp)
        mlflow.log_params(vars(artifacts["params"]))

    # Save for repo
    with open(Path(model_dir, "params.json"), "w") as fp:
        json.dump(vars(params), fp=fp, indent=2, cls=NumpyEncoder)
    artifacts["label_encoder"].save(Path(model_dir, "label_encoder.json"))
    artifacts["tokenizer"].save(Path(model_dir, "tokenizer.json"))
    torch.save(artifacts["model"].state_dict(), Path(model_dir, "model.pt"))
    utils.save_dict(performance, Path(model_dir, "performance.json"))


@app.command()
def predict_tags(
    text: Optional[str] = "Transfer learning with BERT for self-supervised learning",
    model_dir: Path = config.MODEL_DIR,
) -> Dict:
    """Predict tags for a give input text using a trained model.

    Warning:
        Make sure that you have a trained model first!

    Args:
        text (str, optional): Input text to predict tags for.
                              Defaults to "Transfer learning with BERT for self-supervised learning".
        model_dir (Path): location of model artifacts. Defaults to config.MODEL_DIR.

    Raises:
        ValueError: Run id doesn't exist in experiment.

    Returns:
        Predicted tags for input text.
    """
    # Predict
    artifacts = main.load_artifacts(model_dir=model_dir)
    prediction = predict.predict(texts=[text], artifacts=artifacts)
    logger.info(json.dumps(prediction, indent=2))

    return prediction


@app.command()
def diff(commit_a: str = "workspace", commit_b: str = "head"):  # pragma: no cover
    """Compare relevant differences (params, metrics) between commits.
    Inspired by DVC's `dvc metrics diff`but repurposed to
    display diffs pertinent to our experiments.

    Args:
        commit_a (str, optional): Primary commit. Defaults to "workspace".
        commit_b (str, optional): Commit to compare to. Defaults to "head".

    Raises:
        ValueError: Invalid commit.
    """
    diffs = {}
    commits = ["a", "b"]
    if commit_a.lower() in ("head", "current"):
        commit_a = "main"
    if commit_b.lower() in ("head", "current"):
        commit_b = "main"

    # Get params
    params = {"a": {}, "b": {}}
    for i, commit in enumerate([commit_a, commit_b]):
        if commit == "workspace":
            params[commits[i]] = utils.load_dict(filepath=Path(config.CONFIG_DIR, "params.json"))
            continue
        params_url = (
            f"https://raw.githubusercontent.com/GokuMohandas/applied-ml/{commit}/model/params.json"
        )
        params[commits[i]] = utils.load_json_from_url(url=params_url)

    # Parameter differences
    diffs["params"] = {}
    for arg in params["a"]:
        a = params["a"][arg]
        b = params["b"][arg]
        if a != b:
            diffs["params"][arg] = {commit_a: a, commit_b: b}
    logger.info(f"Parameter differences:\n{json.dumps(diffs['params'], indent=2)}")

    # Get metrics
    metrics = {"a": {}, "b": {}}
    for i, commit in enumerate([commit_a, commit_b]):
        if commit == "workspace":
            metrics[commits[i]] = utils.load_dict(
                filepath=Path(config.MODEL_DIR, "performance.json")
            )
            continue
        metrics_url = f"https://raw.githubusercontent.com/GokuMohandas/applied-ml/{commit}/model/performance.json"
        metrics[commits[i]] = utils.load_json_from_url(url=metrics_url)

    # Recursively flatten
    metrics_a = pd.json_normalize(metrics["a"], sep=".").to_dict(orient="records")[0]
    metrics_b = pd.json_normalize(metrics["b"], sep=".").to_dict(orient="records")[0]
    if metrics_a.keys() != metrics_b.keys():
        raise Exception("Cannot compare these commits because they have different metrics.")

    # Metric differences
    diffs["metrics"] = {}
    diffs["metrics"]["improvements"] = {}
    diffs["metrics"]["regressions"] = {}
    for metric in metrics_a:
        if (
            (metric in metrics_b)
            and (metrics_a[metric] != metrics_b[metric])
            and (isinstance(metrics_a[metric], numbers.Number))
            and (metric.split(".")[-1] != "num_samples")
        ):
            item = {
                commit_a: metrics_a[metric],
                commit_b: metrics_b[metric],
                "diff": metrics_a[metric] - metrics_b[metric],
            }
            if item["diff"] >= 0.0:
                diffs["metrics"]["improvements"][metric] = item
            else:
                diffs["metrics"]["regressions"][metric] = item
    logger.info(f"Metric differences:\n{json.dumps(diffs['metrics'], indent=2)}")

    return diffs


@app.command()
def behavioral_reevaluation(
    model_dir: Path = config.MODEL_DIR,
):  # pragma: no cover, requires changing existing runs
    """Reevaluate existing runs on current behavioral tests in eval.py.
    This is possible since behavioral tests are inputs applied to black box
    models and compared with expected outputs. There is not dependency on
    data or model versions.

    Args:
        model_dir (Path): location of model artifacts. Defaults to config.MODEL_DIR.

    Raises:
        ValueError: Run id doesn't exist in experiment.
    """

    # Generate behavioral report
    artifacts = main.load_artifacts(model_dir=model_dir)
    artifacts["performance"]["behavioral"] = eval.get_behavioral_report(artifacts=artifacts)
    mlflow.log_metric("behavioral_score", artifacts["performance"]["behavioral"]["score"])

    # Log updated performance
    utils.save_dict(artifacts["performance"], Path(model_dir, "performance.json"))
