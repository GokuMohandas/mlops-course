# tests/app/test_cli.py
# Test app/cli.py components.

import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

from app.cli import app
from tagifai import config, utils

runner = CliRunner()


def test_download_data():
    result = runner.invoke(app, ["download-data"])
    assert result.exit_code == 0
    assert "Data downloaded!" in result.stdout


@pytest.mark.training
def test_optimize():
    study_name = "test_optimization"
    result = runner.invoke(
        app,
        [
            "optimize",
            "--params-fp",
            f"{Path(config.CONFIG_DIR, 'test_params.json')}",
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
    shutil.rmtree(Path(config.MODEL_REGISTRY, ".trash"))


@pytest.mark.training
def test_train_model():
    experiment_name = "test_experiment"
    run_name = "test_run"
    tmp_dir = Path(config.BASE_DIR, "tmp")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    result = runner.invoke(
        app,
        [
            "train-model",
            "--params-fp",
            f"{Path(config.CONFIG_DIR, 'test_params.json')}",
            "--model-dir",
            f"{tmp_dir}",
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
    shutil.rmtree(Path(config.MODEL_REGISTRY, ".trash"))
    shutil.rmtree(tmp_dir)


def test_predict_tags():
    result = runner.invoke(app, ["predict-tags", "--text", "Transfer learning with BERT."])
    assert result.exit_code == 0
    assert "predicted_tags" in result.stdout


def test_params():
    result = runner.invoke(app, ["params"])
    assert result.exit_code == 0
    assert "seed" in result.stdout


def test_performance():
    result = runner.invoke(app, ["performance"])
    assert result.exit_code == 0
    assert "overall" in result.stdout
