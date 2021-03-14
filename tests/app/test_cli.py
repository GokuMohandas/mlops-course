# tests/app/test_cli.py
# Test app/cli.py components.

import shutil
from pathlib import Path

import mlflow
import pytest
import yaml
from typer.testing import CliRunner

from app.cli import app
from tagifai import config, utils

runner = CliRunner()


def test_download_data():
    result = runner.invoke(app, ["download-data"])
    assert result.exit_code == 0
    assert "Data downloaded!" in result.stdout


@pytest.mark.training
def test_train_model():
    experiment_name = "test_experiment"
    run_name = "test_run"
    result = runner.invoke(
        app,
        [
            "train-model",
            "--args-fp",
            f"{Path(config.CONFIG_DIR, 'test_args.json')}",
            "--experiment-name",
            f"{experiment_name}",
            "--run-name",
            f"{run_name}",
        ],
    )
    assert result.exit_code == 0
    assert "Epoch: 1" in result.stdout
    assert "f1" in result.stdout

    # Delete experiment
    utils.delete_experiment(experiment_name=experiment_name)
    shutil.rmtree(Path(config.EXPERIMENTS_DIR, ".trash"))


def test_predict_tags():
    result = runner.invoke(app, ["predict-tags", "--text", "Transfer learning with BERT."])
    assert result.exit_code == 0
    assert "predicted_tags" in result.stdout


@pytest.mark.training
def test_optimize():
    study_name = "test_optimization"
    result = runner.invoke(
        app,
        [
            "optimize",
            "--args-fp",
            f"{Path(config.CONFIG_DIR, 'test_args.json')}",
            "--study-name",
            f"{study_name}",
            "--num-trials",
            1,
        ],
    )
    assert result.exit_code == 0
    assert "Trial 0" in result.stdout

    # Delete study
    utils.delete_experiment(experiment_name=study_name)
    shutil.rmtree(Path(config.EXPERIMENTS_DIR, ".trash"))


def test_get_sorted_runs():
    result = runner.invoke(app, ["get-sorted-runs"])
    assert result.exit_code == 0
    assert "run_id" in result.stdout


def test_fix_artifact_metadata():
    runner.invoke(app, ["fix-artifact-metadata"])

    # Check an experiment
    sample_meta_yaml = list(Path(config.EXPERIMENTS_DIR).glob("*/meta.yaml"))[0]
    with open(sample_meta_yaml) as f:
        metadata = yaml.load(f)
        experiment_id = metadata["artifact_location"].split("/")[-1]
        expected_artifact_location = Path("file://", config.EXPERIMENTS_DIR, experiment_id)
        assert metadata["artifact_location"] == str(expected_artifact_location)

    # Check a run
    sample_meta_yaml = list(Path(config.EXPERIMENTS_DIR).glob("*/*/meta.yaml"))[0]
    with open(sample_meta_yaml) as f:
        metadata = yaml.load(f)
        experiment_id = metadata["artifact_uri"].split("/")[-3]
        run_id = metadata["artifact_uri"].split("/")[-2]
        expected_artifact_uri = Path(
            "file://",
            config.EXPERIMENTS_DIR,
            experiment_id,
            run_id,
            "artifacts",
        )
        assert metadata["artifact_uri"] == str(expected_artifact_uri)


def test_clean_experiments():
    # Keep `best` experiment (default)
    result = runner.invoke(app, ["clean-experiments", "--experiments-to-keep", "best"])
    assert result.exit_code == 0
    client = mlflow.tracking.MlflowClient()
    experiments = client.list_experiments()
    assert len(experiments) == 1
    assert experiments[0].name == "best"

    # Must keep at least one experiment
    with pytest.raises(ValueError):
        result = runner.invoke(app, ["clean-experiments", "--experiments-to-keep", ""])
        assert result.exit_code == 1
        raise result.exc_info[1]
