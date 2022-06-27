import numpy as np
import pandas as pd
import pytest
from snorkel.slicing import PandasSFApplier, slice_dataframe

from tagifai import evaluate


@pytest.fixture(scope="module")
def df():
    data = [
        {"text": "CNNs for text processing.", "tag": "natural-language-processing"},
        {"text": "This is short text.", "tag": "other"},
        {"text": "This is a very very very very long text.", "tag": "other"},
    ]
    df = pd.DataFrame(data)
    return df


@pytest.mark.parametrize(
    "f, indices",
    [(evaluate.nlp_cnn, [0]), (evaluate.short_text, [0, 1])],
)
def test_slice_functions(df, f, indices):
    assert slice_dataframe(df, f).index.tolist() == indices


def test_get_slices_metrics(df):
    y_true = np.array([0, 1, 1])
    y_pred = np.array([0, 0, 1])
    slices = PandasSFApplier([evaluate.nlp_cnn, evaluate.short_text]).apply(df)
    metrics = evaluate.get_slice_metrics(y_true=y_true, y_pred=y_pred, slices=slices)
    assert metrics["nlp_cnn"]["precision"] == 1 / 1
    assert metrics["nlp_cnn"]["recall"] == 1 / 1
    assert metrics["short_text"]["precision"] == 1 / 2
    assert metrics["short_text"]["recall"] == 1 / 2


def test_get_metrics():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 1])
    classes = ["a", "b"]
    performance = evaluate.get_metrics(y_true=y_true, y_pred=y_pred, classes=classes, df=None)
    assert performance["overall"]["precision"] == 2 / 4
    assert performance["overall"]["recall"] == 2 / 4
    assert performance["class"]["a"]["precision"] == 1 / 2
    assert performance["class"]["a"]["recall"] == 1 / 2
    assert performance["class"]["b"]["precision"] == 1 / 2
    assert performance["class"]["b"]["recall"] == 1 / 2
