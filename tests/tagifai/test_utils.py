# tests/tagifai/test_utils.py
# Test tagifai/utils.py components.

import tempfile
from pathlib import Path

import numpy as np
import pytest

from tagifai import utils


def test_load_json_from_url():
    tags_url = "https://raw.githubusercontent.com/GokuMohandas/MadeWithML/main/datasets/tags.json"
    tags_dict = utils.list_to_dict(utils.load_json_from_url(url=tags_url), key="tag")
    assert "transformers" in tags_dict


def test_save_and_load_dict():
    with tempfile.TemporaryDirectory() as dp:
        d = {"hello": "world"}
        fp = Path(dp, "d.json")
        utils.save_dict(d=d, filepath=fp)
        d = utils.load_dict(filepath=fp)
        assert d["hello"] == "world"


def test_list_to_dict():
    list_of_dicts = [{"tag": "attention", "parents": []}, {"tag": "bert", "parents": "transformer"}]
    d = utils.list_to_dict(list_of_dicts=list_of_dicts, key="tag")
    assert isinstance(d, dict)
    assert isinstance(d["attention"], dict)
    assert "tag" not in d["attention"]
    assert list(d.keys()) == ["attention", "bert"]
    assert d["bert"]["parents"] == "transformer"


def test_set_seed():
    utils.set_seed()
    a = np.random.randn(2, 3)
    b = np.random.randn(2, 3)
    utils.set_seed()
    x = np.random.randn(2, 3)
    y = np.random.randn(2, 3)
    assert np.array_equal(a, x)
    assert np.array_equal(b, y)


@pytest.mark.parametrize(
    "d_a, d_b, diff",
    [
        (
            {"v1": 1, "v2": 1},
            {"v1": 1, "v2": 1},
            {"v1": {"a": 1, "b": 1, "diff": 0}, "v2": {"a": 1, "b": 1, "diff": 0}},
        ),  # no diff
        (
            {"v1": 3, "v2": 3},
            {"v1": 1, "v2": 1},
            {"v1": {"a": 3, "b": 1, "diff": 2}, "v2": {"a": 3, "b": 1, "diff": 2}},
        ),  # diff
        (
            {"v1": 3, "v2": "word"},
            {"v1": 1, "v2": "word"},
            {"v1": {"a": 3, "b": 1, "diff": 2}},
        ),  # one numerical
        ({"v1": "word", "v2": "word"}, {"v1": "word", "v2": "word"}, {}),  # no numerical
        (
            {"v1": {"v2": 3}, "v3": 3},
            {"v1": {"v2": 1}, "v3": 1},
            {"v1.v2": {"a": 3, "b": 1, "diff": 2}, "v3": {"a": 3, "b": 1, "diff": 2}},
        ),  # nested
    ],
)
def test_dict_diff(d_a, d_b, diff):
    assert utils.dict_diff(d_a=d_a, d_b=d_b) == diff


@pytest.mark.parametrize(
    "d_a, d_b, exception",
    [
        ({"v1": 1, "v2": 1}, {"v1": 1, "v3": 1}, Exception),
    ],
)
def test_dict_diff_exception(d_a, d_b, exception):
    with pytest.raises(exception):
        utils.dict_diff(d_a=d_a, d_b=d_b)
