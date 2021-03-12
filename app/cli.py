# cli.py
# Command line interface (CLI) application.

import json
import shutil
import tempfile
import warnings
from argparse import Namespace
from pathlib import Path
from typing import Dict, Optional

import mlflow
import optuna
import torch
import typer
import yaml
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
def train_model(
    args_fp: Path = Path(config.CONFIG_DIR, "args.json"),
    experiment_name: Optional[str] = "best",
    run_name: Optional[str] = "model",
    publish_metrics: Optional[bool] = False,
    append: Optional[bool] = False,
) -> None:
    """Train a model using the specified parameters.

    Args:
        args_fp (Path, optional): Location of arguments to use for training. Defaults to `config/args.json`.
        experiment_name (str, optional): Name of the experiment to save the run to. Defaults to `best`.
        run_name (str, optional): Name of the run. Defaults to `model`.
    """
    # Set experiment and start run
    args = Namespace(**utils.load_dict(filepath=args_fp))

    # Ensure experiment doesn't have a preexisting run
    runs = utils.get_sorted_runs(
        experiment_name=experiment_name,
        order_by=["metrics.f1 DESC"],
        verbose=False,
    )
    if len(runs) and not append:
        raise Exception(
            f"You already have an existing run for Experiment {experiment_name}.\n"
            "If you'd like to append a run to this experiment, rerun this command with the --append flag."
        )

    # Start run
    mlflow.set_experiment(experiment_name=experiment_name)
    with mlflow.start_run(run_name=run_name):
        # Train
        artifacts = main.run(args=args)

        # Set tags
        tags = {}
        mlflow.set_tags(tags)

        # Log metrics
        performance = artifacts["performance"]
        behavioral_report = artifacts["behavioral_report"]
        logger.info(json.dumps(performance["overall"], indent=2))
        metrics = {
            "precision": performance["overall"]["precision"],
            "recall": performance["overall"]["recall"],
            "f1": performance["overall"]["f1"],
            "best_val_loss": artifacts["loss"],
            "behavioral_score": behavioral_report["score"],
            "slices_f1": performance["slices"]["f1"],
        }
        mlflow.log_metrics(metrics)

        # Log artifacts
        with tempfile.TemporaryDirectory() as dp:
            artifacts["label_encoder"].save(Path(dp, "label_encoder.json"))
            artifacts["tokenizer"].save(Path(dp, "tokenizer.json"))
            torch.save(artifacts["model"].state_dict(), Path(dp, "model.pt"))
            utils.save_dict(performance, Path(dp, "performance.json"))
            utils.save_dict(behavioral_report, Path(dp, "behavioral_report.json"))
            mlflow.log_artifacts(dp)
        mlflow.log_params(vars(artifacts["args"]))

        # Publish metrics
        if publish_metrics:  # pragma: no cover, boolean to publish metrics
            utils.save_dict(performance, Path(config.METRICS_DIR, "performance.json"))
            utils.save_dict(behavioral_report, Path(config.METRICS_DIR, "behavioral_report.json"))


@app.command()
def predict_tags(
    text: Optional[str] = "Transfer learning with BERT for self-supervised learning",
    experiment_name: Optional[str] = "best",
    run_id: Optional[str] = "",
) -> Dict:
    """Predict tags for a give input text using a trained model.

    Warning:
        Make sure that you have a trained model first!

    Args:
        text (str, optional): Input text to predict tags for.
                              Defaults to "Transfer learning with BERT for self-supervised learning".
        experiment_name (str, optional): Name of the experiment to fetch run from.
        run_id (str, optional): ID of the run to load model artifacts from.
                                Defaults to run with highest F1 score.

    Raises:
        ValueError: Run id doesn't exist in experiment.

    Returns:
        Predicted tags for input text.
    """
    # Get sorted runs
    runs = utils.get_sorted_runs(
        experiment_name=experiment_name,
        order_by=["metrics.f1 DESC"],
    )
    run_ids = [run["run_id"] for run in runs]

    # Get best run
    if not run_id:
        run_id = run_ids[0]

    # Validate run id
    if run_id not in run_ids:  # pragma: no cover, simple value check
        raise ValueError(f"Run_id {run_id} does not exist in experiment {experiment_name}")

    # Predict
    artifacts = main.load_artifacts(run_id=run_id)
    prediction = predict.predict(texts=[text], artifacts=artifacts)
    logger.info(json.dumps(prediction, indent=2))

    return prediction


@app.command()
def optimize(
    args_fp: Path = Path(config.CONFIG_DIR, "args.json"),
    study_name: Optional[str] = "optimization",
    num_trials: int = 100,
) -> None:
    """Optimize a subset of hyperparameters towards an objective.

    This saves the best trial's arguments into `config/args.json`.

    Args:
        args_fp (Path, optional): Location of arguments (just using num_samples,
                                  num_epochs, etc.) to use for training.
                                  Defaults to `config/args.json`.
        study_name (str, optional): Name of the study to save trial runs under. Defaults to `optimization`.
        num_trials (int, optional): Number of trials to run. Defaults to 100.
    """
    # Starting arguments (not actually used but needed for set up)
    args = Namespace(**utils.load_dict(filepath=args_fp))

    # Optimize
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    study = optuna.create_study(study_name=study_name, direction="maximize", pruner=pruner)
    mlflow_callback = MLflowCallback(tracking_uri=mlflow.get_tracking_uri(), metric_name="f1")
    study.optimize(
        lambda trial: main.objective(args, trial),
        n_trials=num_trials,
        callbacks=[mlflow_callback],
    )

    # All trials
    trials_df = study.trials_dataframe()
    trials_df = trials_df.sort_values(["value"], ascending=False)

    # Best trial
    logger.info(f"Best value (f1): {study.best_trial.value}")
    params = {**args.__dict__, **study.best_trial.params}
    params["threshold"] = study.best_trial.user_attrs["threshold"]
    with open(args_fp, "w") as fp:
        json.dump(params, fp=fp, indent=2, cls=NumpyEncoder)
    logger.info(json.dumps(params, indent=2, cls=NumpyEncoder))


@app.command()
def behavioral_reevaluation(
    experiment_name: Optional[str] = "best",
):  # pragma: no cover, requires changing existing runs
    """Reevaluate existing runs on current behavioral tests in eval.py.
    This is possible since behavioral tests are inputs applied to black box
    models and compared with expected outputs. There is not dependency on
    data or model versions.

    Args:
        experiment_name (str, optional): Name of the experiment to fetch run from.

    Raises:
        ValueError: Run id doesn't exist in experiment.
    """

    def update_behavioral_report(run_id):
        with mlflow.start_run(run_id=run_id):
            # Generate behavioral report
            artifacts = main.load_artifacts(run_id=run_id)
            behavioral_report = eval.get_behavioral_report(artifacts=artifacts)
            mlflow.log_metric("behavioral_score", behavioral_report["score"])

            # Log artifacts
            with tempfile.TemporaryDirectory() as dp:
                utils.save_dict(behavioral_report, Path(dp, "behavioral_report.json"))
                mlflow.log_artifacts(dp)

            # Publish metrics
            utils.save_dict(behavioral_report, Path(config.METRICS_DIR, "behavioral_report.json"))

        logger.info(f"Updated behavioral report for run_id {run_id}")

    # Get sorted runs
    runs = utils.get_sorted_runs(
        experiment_name=experiment_name,
        order_by=["metrics.f1 DESC"],
        verbose=False,
    )
    run_ids = [run["run_id"] for run in runs]

    # Reevaluate behavioral tests for all runs
    for run_id in run_ids:
        update_behavioral_report(run_id=run_id)


@app.command()
def get_sorted_runs(experiment_name: Optional[str] = "best"):
    """Get sorted runs for an experiment."""
    utils.get_sorted_runs(experiment_name=experiment_name, order_by=["metrics.f1 DESC"])


@app.command()
def fix_artifact_metadata():
    """Set the artifact URI for all experiments and runs.
    Used when transferring experiments from other locations (ex. Colab).

    Note:
        check out the [optimize.ipynb](https://colab.research.google.com/github/GokuMohandas/applied-ml/blob/main/notebooks/optimize.ipynb){:target="_blank"} notebook for how to train on Google Colab and transfer to local.
    """

    def fix_artifact_location(fp):
        """Set variable's yaml value on file at fp."""
        with open(fp) as f:
            metadata = yaml.load(f)

        # Set new values
        experiment_id = str(fp).split("/")[-2]
        artifact_location = Path("file://", config.EXPERIMENTS_DIR, experiment_id)
        metadata["artifact_location"] = str(artifact_location)
        metadata["experiment_id"] = experiment_id

        with open(fp, "w") as f:
            yaml.dump(metadata, f)

    def fix_artifact_uri(fp):
        """Set variable's yaml value on file at fp."""
        with open(fp) as f:
            metadata = yaml.load(f)

        # Set new value
        experiment_id = str(fp).split("/")[-3]
        run_id = metadata["run_id"]
        artifact_uri = Path(
            "file://",
            config.EXPERIMENTS_DIR,
            experiment_id,
            run_id,
            "artifacts",
        )
        metadata["experiment_id"] = experiment_id
        metadata["artifact_uri"] = str(artifact_uri)

        with open(fp, "w") as f:
            yaml.dump(metadata, f)

    # Get artifact location
    experiment_meta_yamls = list(Path(config.EXPERIMENTS_DIR).glob("*/meta.yaml"))
    for meta_yaml in experiment_meta_yamls:
        fix_artifact_location(fp=meta_yaml)
        logger.info(f"Set artifact location for {meta_yaml}")

    # Change artifact URI
    run_meta_yamls = list(Path(config.EXPERIMENTS_DIR).glob("*/*/meta.yaml"))
    for meta_yaml in run_meta_yamls:
        fix_artifact_uri(fp=meta_yaml)
        logger.info(f"Set artifact URI for {meta_yaml}")


@app.command()
def clean_experiments(experiments_to_keep: str = "best"):
    """Removes all experiments besides the
    ones specified in `experiments_to_keep`.

    Args:
        experiments_to_keep (str): comma separated string of experiments to keep.
    """
    # Get experiments to keep
    experiments_to_keep = list(set([exp.strip() for exp in experiments_to_keep.split(",")]))
    if not len(experiments_to_keep) or not experiments_to_keep[0]:
        raise ValueError("You must keep at least one experiment.")

    # Filter and delete
    client = mlflow.tracking.MlflowClient()
    for experiment in client.list_experiments():
        if experiment.name not in experiments_to_keep:  # pragma: no cover, mlflow function
            logger.info(f"Deleting Experiment {experiment.name}")
            client.delete_experiment(experiment_id=experiment.experiment_id)

    # Delete MLFlow trash
    shutil.rmtree(Path(config.EXPERIMENTS_DIR, ".trash"))
    logger.info(f"Cleared experiments besides {experiments_to_keep}")
