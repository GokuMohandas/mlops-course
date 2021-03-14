# tests/tagifai/test_eval.py
# Test tagifai/eval.py components.

import numpy as np

from tagifai import eval


def test_get_metrics():
    y_true = np.array([[1, 0], [0, 1], [0, 0], [0, 0], [1, 1]])
    y_pred = np.array([[0, 1], [0, 1], [0, 1], [0, 1], [1, 1]])
    classes = ["a", "b"]
    performance = eval.get_metrics(y_true=y_true, y_pred=y_pred, classes=classes, df=None)
    assert performance["overall"]["precision"] == (1 / 1 + 2 / 5) / 2
    assert performance["overall"]["recall"] == (1 / 2 + 2 / 2) / 2
    assert performance["class"]["a"]["precision"] == 1 / 1
    assert performance["class"]["a"]["recall"] == 1 / 2
    assert performance["class"]["b"]["precision"] == 2 / 5
    assert performance["class"]["b"]["recall"] == 2 / 2
