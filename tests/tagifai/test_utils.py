# tests/tagifai/test_utils.py
# Test tagifai/utils.py components.

import tempfile
from pathlib import Path

import numpy as np

from tagifai import utils


def test_load_json_from_url():
    tags_url = "https://raw.githubusercontent.com/GokuMohandas/madewithml/main/datasets/tags.json"
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
