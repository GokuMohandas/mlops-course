import json
import tempfile
import warnings
from argparse import Namespace
from pathlib import Path
from typing import Dict

import joblib
import mlflow
import optuna
import pandas as pd
import typer
from numpyencoder import NumpyEncoder
from optuna.integration.mlflow import MLflowCallback

from config import config
from config.config import logger
from tagifai import data, predict, train, utils

warnings.filterwarnings("ignore")

# Initialize Typer CLI app
app = typer.Typer()


@app.command()
def etl_data():
    """Extract, load and transform our data assets."""
    # Extract
    projects = utils.load_json_from_url(url=config.PROJECTS_URL)
    tags = utils.load_json_from_url(url=config.TAGS_URL)

    # Transform
    df = pd.DataFrame(projects)
    df = df[df.tag.notnull()]  # drop rows w/ no tag

    # Load
    projects_fp = Path(config.DATA_DIR, "projects.json")
    utils.save_dict(d=df.to_dict(orient="records"), filepath=projects_fp)
    tags_fp = Path(config.DATA_DIR, "tags.json")
    utils.save_dict(d=tags, filepath=tags_fp)

    logger.info("✅ ETL on data is complete!")


@app.command()
def label_data(args_fp: str = "config/args.json") -> None:
    """Label data with constraints.

    Args:
        args_fp (str): location of args.
    """
    # Load projects
    projects_fp = Path(config.DATA_DIR, "projects.json")
    projects = utils.load_dict(filepath=projects_fp)
    df = pd.DataFrame(projects)

    # Load tags
    tags_dict = {}
    tags_fp = Path(config.DATA_DIR, "tags.json")
    for item in utils.load_dict(filepath=tags_fp):
        key = item.pop("tag")
        tags_dict[key] = item

    # Label with constrains
    args = Namespace(**utils.load_dict(filepath=args_fp))
    df = df[df.tag.notnull()]  # remove projects with no label
    df = data.replace_oos_labels(df=df, labels=tags_dict.keys(), label_col="tag", oos_label="other")
    df = data.replace_minority_labels(
        df=df, label_col="tag", min_freq=args.min_freq, new_label="other"
    )

    # Save clean labeled data
    labeled_projects_fp = Path(config.DATA_DIR, "labeled_projects.json")
    utils.save_dict(d=df.to_dict(orient="records"), filepath=labeled_projects_fp)
    logger.info("✅ Saved labeled data!")


@app.command()
def train_model(
    args_fp: str = "config/args.json",
    experiment_name: str = "baselines",
    run_name: str = "sgd",
    test_run: bool = False,
) -> None:
    """Train a model given arguments.

    Args:
        args_fp (str): location of args.
        experiment_name (str): name of experiment.
        run_name (str): name of specific run in experiment.
        test_run (bool, optional): If True, artifacts will not be saved. Defaults to False.
    """
    # Load labeled data
    projects_fp = Path(config.DATA_DIR, "labeled_projects.json")
    projects = utils.load_dict(filepath=projects_fp)
    df = pd.DataFrame(projects)

    # Train
    args = Namespace(**utils.load_dict(filepath=args_fp))
    mlflow.set_experiment(experiment_name=experiment_name)
    with mlflow.start_run(run_name=run_name):
        run_id = mlflow.active_run().info.run_id
        logger.info(f"Run ID: {run_id}")
        artifacts = train.train(df=df, args=args)
        performance = artifacts["performance"]
        logger.info(json.dumps(performance, indent=2))

        # Log metrics and parameters
        performance = artifacts["performance"]
        mlflow.log_metrics({"precision": performance["overall"]["precision"]})
        mlflow.log_metrics({"recall": performance["overall"]["recall"]})
        mlflow.log_metrics({"f1": performance["overall"]["f1"]})
        mlflow.log_params(vars(artifacts["args"]))

        # Log artifacts
        with tempfile.TemporaryDirectory() as dp:
            utils.save_dict(vars(artifacts["args"]), Path(dp, "args.json"), cls=NumpyEncoder)
            artifacts["label_encoder"].save(Path(dp, "label_encoder.json"))
            joblib.dump(artifacts["vectorizer"], Path(dp, "vectorizer.pkl"))
            joblib.dump(artifacts["model"], Path(dp, "model.pkl"))
            utils.save_dict(performance, Path(dp, "performance.json"))
            mlflow.log_artifacts(dp)

    # Save to config
    if not test_run:  # pragma: no cover, actual run
        open(Path(config.CONFIG_DIR, "run_id.txt"), "w").write(run_id)
        utils.save_dict(performance, Path(config.CONFIG_DIR, "performance.json"))


@app.command()
def optimize(
    args_fp: str = "config/args.json", study_name: str = "optimization", num_trials: int = 20
) -> None:
    """Optimize hyperparameters.

    Args:
        args_fp (str): location of args.
        study_name (str): name of optimization study.
        num_trials (int): number of trials to run in study.
    """
    # Load labeled data
    projects_fp = Path(config.DATA_DIR, "labeled_projects.json")
    projects = utils.load_dict(filepath=projects_fp)
    df = pd.DataFrame(projects)

    # Optimize
    args = Namespace(**utils.load_dict(filepath=args_fp))
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    study = optuna.create_study(study_name=study_name, direction="maximize", pruner=pruner)
    mlflow_callback = MLflowCallback(tracking_uri=mlflow.get_tracking_uri(), metric_name="f1")
    study.optimize(
        lambda trial: train.objective(args, df, trial),
        n_trials=num_trials,
        callbacks=[mlflow_callback],
    )

    # Best trial
    trials_df = study.trials_dataframe()
    trials_df = trials_df.sort_values(["user_attrs_f1"], ascending=False)
    args = {**args.__dict__, **study.best_trial.params}
    utils.save_dict(d=args, filepath=args_fp, cls=NumpyEncoder)
    logger.info(f"\nBest value (f1): {study.best_trial.value}")
    logger.info(f"Best hyperparameters: {json.dumps(study.best_trial.params, indent=2)}")


def load_artifacts(run_id: str = None) -> Dict:
    """Load artifacts for a given run_id.

    Args:
        run_id (str): id of run to load artifacts from.

    Returns:
        Dict: run's artifacts.
    """
    if not run_id:
        run_id = open(Path(config.CONFIG_DIR, "run_id.txt")).read()

    # Locate specifics artifacts directory
    experiment_id = mlflow.get_run(run_id=run_id).info.experiment_id
    artifacts_dir = Path(config.MODEL_REGISTRY, experiment_id, run_id, "artifacts")

    # Load objects from run
    args = Namespace(**utils.load_dict(filepath=Path(artifacts_dir, "args.json")))
    vectorizer = joblib.load(Path(artifacts_dir, "vectorizer.pkl"))
    label_encoder = data.LabelEncoder.load(fp=Path(artifacts_dir, "label_encoder.json"))
    model = joblib.load(Path(artifacts_dir, "model.pkl"))
    performance = utils.load_dict(filepath=Path(artifacts_dir, "performance.json"))

    return {
        "args": args,
        "label_encoder": label_encoder,
        "vectorizer": vectorizer,
        "model": model,
        "performance": performance,
    }


@app.command()
def predict_tag(text: str = "", run_id: str = None) -> None:
    """Predict tag for text.

    Args:
        text (str): input text to predict label for.
        run_id (str, optional): run id to load artifacts for prediction. Defaults to None.
    """
    if not run_id:
        run_id = open(Path(config.CONFIG_DIR, "run_id.txt")).read()
    artifacts = load_artifacts(run_id=run_id)
    prediction = predict.predict(texts=[text], artifacts=artifacts)
    logger.info(json.dumps(prediction, indent=2))
    return prediction


if __name__ == "__main__":
    app()  # pragma: no cover, live app
