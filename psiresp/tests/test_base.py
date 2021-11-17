import pytest
import numpy as np

from psiresp.base import _to_immutable


@pytest.mark.parametrize("object_in, object_out", [
    (np.ones(3), (1, 1, 1)),
    (np.zeros((2, 2)), ((0, 0), (0, 0))),  # recursive list
    ({1, 2, 3}, frozenset({1, 2, 3})),
    (
        [{'key': {'key3': 3, 'key2': {1, 2}}}],
        ((('key', (('key2', frozenset({1, 2})), ('key3', 3))),),)
    )
])
def test_to_immutable(object_in, object_out):
    immuted = _to_immutable(object_in)
    assert immuted == object_out
    hash(immuted)
