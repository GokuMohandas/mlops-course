import numpy as np
import pytest

from tagifai import predict


@pytest.mark.parametrize(
    "threshold, y_pred",
    [
        (0.5, [0]),
        (0.6, [1]),
        (0.75, [1]),
    ],
)
def test_custom_predict(threshold, y_pred):
    y_prob = np.array([[0.6, 0.4]])
    assert predict.custom_predict(y_prob=y_prob, threshold=threshold, index=1) == y_pred
