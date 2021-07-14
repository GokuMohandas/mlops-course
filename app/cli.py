# app/cli.py
# Command line interface (CLI) application.

import json
import tempfile
import warnings
from argparse import Namespace
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import mlflow
import optuna
import pandas as pd
import torch
import typer
from feast import FeatureStore
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
    """Load data from URL and save to local drive."""
    # Download data
    projects_url = (
        "https://raw.githubusercontent.com/GokuMohandas/MadeWithML/main/datasets/projects.json"
    )
    tags_url = "https://raw.githubusercontent.com/GokuMohandas/MadeWithML/main/datasets/tags.json"
    projects = utils.load_json_from_url(url=projects_url)
    tags = utils.load_json_from_url(url=tags_url)

    # Save data
    projects_fp = Path(config.DATA_DIR, "projects.json")
    tags_fp = Path(config.DATA_DIR, "tags.json")
    utils.save_dict(d=projects, filepath=projects_fp)
    utils.save_dict(d=tags, filepath=tags_fp)
    logger.info("✅ Data downloaded!")


@app.command()
def compute_features(
    params_fp: Path = Path(config.CONFIG_DIR, "params.json"),
) -> None:
    """Compute and save features for training.

    Args:
        params_fp (Path, optional): Location of parameters (just using num_samples,
                                    num_epochs, etc.) to use for training.
                                    Defaults to `config/params.json`.
    """
    # Parameters
    params = Namespace(**utils.load_dict(filepath=params_fp))

    # Compute features
    main.compute_features(params=params)


@app.command()
def get_historical_features():
    # Entities to pull data for (should dynamically read this from somewhere)
    project_ids = [1, 2, 3]
    now = datetime.now()
    timestamps = [datetime(now.year, now.month, now.day)] * len(project_ids)
    entity_df = pd.DataFrame.from_dict({"id": project_ids, "event_timestamp": timestamps})

    # Get historical features
    store = FeatureStore(repo_path=Path(config.BASE_DIR, "features"))
    training_df = store.get_historical_features(
        entity_df=entity_df,
        feature_refs=["project_details:text", "project_details:tags"],
    ).to_df()

    # Store in location for training task to pick up
    print(training_df.head())


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
    # Parameters
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
    utils.save_dict(params, params_fp, cls=NumpyEncoder)
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
    # Parameters
    params = Namespace(**utils.load_dict(filepath=params_fp))

    # Start run
    mlflow.set_experiment(experiment_name=experiment_name)
    with mlflow.start_run(run_name=run_name):
        run_id = mlflow.active_run().info.run_id

        # Train
        artifacts = main.train_model(params=params)

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
            utils.save_dict(vars(artifacts["params"]), Path(dp, "params.json"), cls=NumpyEncoder)
            utils.save_dict(performance, Path(dp, "performance.json"))
            artifacts["label_encoder"].save(Path(dp, "label_encoder.json"))
            artifacts["tokenizer"].save(Path(dp, "tokenizer.json"))
            torch.save(artifacts["model"].state_dict(), Path(dp, "model.pt"))
            mlflow.log_artifacts(dp)
        mlflow.log_params(vars(artifacts["params"]))

    # Save for repo
    open(Path(model_dir, "run_id.txt"), "w").write(run_id)
    utils.save_dict(vars(params), Path(model_dir, "params.json"), cls=NumpyEncoder)
    utils.save_dict(performance, Path(model_dir, "performance.json"))


@app.command()
def predict_tags(
    text: Optional[str] = "Transfer learning with BERT for self-supervised learning",
    run_id: str = open(Path(config.MODEL_DIR, "run_id.txt")).read(),
) -> Dict:
    """Predict tags for a give input text using a trained model.

    Warning:
        Make sure that you have a trained model first!

    Args:
        text (str, optional): Input text to predict tags for.
                              Defaults to "Transfer learning with BERT for self-supervised learning".
        run_id (str): ID of the model run to load artifacts. Defaults to run ID in config.MODEL_DIR.

    Raises:
        ValueError: Run id doesn't exist in experiment.

    Returns:
        Predicted tags for input text.
    """
    # Predict
    artifacts = main.load_artifacts(run_id=run_id)
    prediction = predict.predict(texts=[text], artifacts=artifacts)
    logger.info(json.dumps(prediction, indent=2))

    return prediction


@app.command()
def params(
    author: str = config.AUTHOR,
    repo: str = config.REPO,
    tag: str = "workspace",
    verbose: bool = True,
):
    if tag == "workspace":
        params = utils.load_dict(filepath=Path(config.MODEL_DIR, "params.json"))
    else:  # pragma: no cover, project specific condition
        url = f"https://raw.githubusercontent.com/{author}/{repo}/{tag}/model/params.json"
        params = utils.load_json_from_url(url=url)
    if verbose:
        logger.info(json.dumps(params, indent=2))
    return params


@app.command()
def performance(
    author: str = config.AUTHOR,
    repo: str = config.REPO,
    tag: str = "workspace",
    verbose: bool = True,
):
    if tag == "workspace":
        performance = utils.load_dict(filepath=Path(config.MODEL_DIR, "performance.json"))
    else:  # pragma: no cover, project specific condition
        url = f"https://raw.githubusercontent.com/{author}/{repo}/{tag}/model/performance.json"
        performance = utils.load_json_from_url(url=url)
    if verbose:
        logger.info(json.dumps(performance, indent=2))
    return performance


@app.command()
def diff(
    author: str = config.AUTHOR,
    repo: str = config.REPO,
    tag_a: str = "workspace",
    tag_b: str = "",
):  # pragma: no cover, can't be certain what diffs will exist
    # Tag b
    if tag_b == "":
        tags_url = f"https://api.github.com/repos/{author}/{repo}/tags"
        tag_b = utils.load_json_from_url(url=tags_url)[0]["name"]
    logger.info(f"Comparing {tag_a} with {tag_b}:")

    # Params
    params_a = params(author=author, repo=repo, tag=tag_a, verbose=False)
    params_b = params(author=author, repo=repo, tag=tag_b, verbose=False)
    params_diff = utils.dict_diff(d_a=params_a, d_b=params_b, d_a_name=tag_a, d_b_name=tag_b)
    logger.info(f"Parameter differences: {json.dumps(params_diff, indent=2)}")

    # Performance
    performance_a = performance(author=author, repo=repo, tag=tag_a, verbose=False)
    performance_b = performance(author=author, repo=repo, tag=tag_b, verbose=False)
    performance_diff = utils.dict_diff(
        d_a=performance_a, d_b=performance_b, d_a_name=tag_a, d_b_name=tag_b
    )
    logger.info(f"Performance differences: {json.dumps(performance_diff, indent=2)}")

    return params_diff, performance_diff


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
