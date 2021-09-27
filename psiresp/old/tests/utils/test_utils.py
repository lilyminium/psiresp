import pytest

from psiresp.utils import utils


@pytest.mark.parametrize("obj, value", [
    ([1], True),
    (1, False),
    ("asdf", False),
    (map(str, [2, 3]), True),
])
def test_is_iterable(obj, value):
    assert utils.is_iterable(obj) == value


@pytest.mark.parametrize("obj, value", [
    ([1], [1]),
    (1, [1]),
    ("asdf", ["asdf"]),
])
def test_as_iterable(obj, value):
    assert utils.as_iterable(obj) == value
