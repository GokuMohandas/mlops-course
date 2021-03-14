# tests/tagifai/test_train.py
# Test tagifai/train.py unit components.

import numpy as np

from tagifai import train


def test_find_best_threshold():
    y_true = np.array([[1, 0], [0, 1], [0, 0], [1, 1]])
    y_prob = np.array([[0.75, 0.25], [0.25, 0.75], [0.25, 0.25], [0.75, 0.75]])
    assert train.find_best_threshold(y_true=y_true, y_prob=y_prob) == 0.75
