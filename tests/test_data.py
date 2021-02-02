# test_data.py
# Test tagifai/data.py components.


import pytest

from tagifai import data


@pytest.mark.parametrize(
    "items, include, exclude, filtered",
    [
        (["apple"], ["apple"], ["orange"], ["apple"]),
        (
            ["apple", "banana", "grape", "orange"],
            ["apple", "banana", "grape", "orange"],
            [],
            ["apple", "banana", "grape", "orange"],
        ),
        (
            ["apple", "banana", "grape", "orange"],
            [],
            ["apple", "banana", "grape", "orange"],
            [],
        ),
        (
            ["apple", "banana", "grape", "orange"],
            ["apple"],
            ["pineapple"],
            ["apple"],
        ),
    ],
)
def test_filter_items(items, include, exclude, filtered):
    assert (
        data.filter_items(items=items, include=include, exclude=exclude)
        == filtered
    )
