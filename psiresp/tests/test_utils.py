import pytest
import numpy as np

from psiresp.utils import update_dictionary


@pytest.mark.parametrize("update, output", [
    (
        {"key": {"a": 2}},
        {"base": {
            "nested": {"a": 1, "b": 2},
            "key": {"a": 2},
        }, "c": 3},
    ),
    (
        {"nested": {"a": 2}},
        {"base": {
            "nested": {"a": 2, "b": 2},
            "key": "v",
        }, "c": 3},
    ),
    (
        {"nested": {"d": 10}},
        {"base": {
            "nested": {"a": 1, "b": 2, "d": 10},
            "key": "v",
        }, "c": 3},
    ),
    (
        {"nested": {"inner": {"innerkey": "innerv"}}},
        {"base": {
            "nested": {"a": 1, "b": 2, "inner": {"innerkey": "innerv"}},
            "key": "v",
        }, "c": 3},
    ),
])
def test_update_dictionary(update, output):
    base = {"base": {"nested": {"a": 1, "b": 2}, "key": "v"}, "c": 3}
    update_dictionary(base, "base", update)
    assert base == output
