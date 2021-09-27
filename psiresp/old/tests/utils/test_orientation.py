import pytest
from numpy.testing import assert_almost_equal

from psiresp.utils import orientation as orutils

from ..base import coordinates_from_xyzfile
from ..datafiles import (DMSO, DMSO_QMRA,
                         DMSO_O1, DMSO_O2,
                         DMSO_O3, DMSO_O4,
                         )


@pytest.mark.fast
@pytest.mark.parametrize("func, infile, atom_ids, reffile", [
    (orutils.orient_rigid, DMSO, (1, 5, 6), DMSO_O1),
    (orutils.orient_rigid, DMSO, (6, 5, 1), DMSO_O2),
    (orutils.rotate_rigid, DMSO_QMRA, (1, 5, 6), DMSO_O3),
    (orutils.rotate_rigid, DMSO_QMRA, (6, 5, 1), DMSO_O4),
])
def test_rigid_reorientation(func, infile, atom_ids, reffile):
    original = coordinates_from_xyzfile(infile)
    i, j, k = map(lambda x: x-1, atom_ids)
    reoriented = func(i, j, k, original)
    reference = coordinates_from_xyzfile(reffile)
    assert_almost_equal(reoriented, reference, decimal=5)


def test_generate_atom_combinations():
    expected = [(1, 5, 6), (6, 5, 1),
                (1, 5, 7), (7, 5, 1),
                (1, 6, 7), (7, 6, 1),
                (5, 6, 7), (7, 6, 5),
                # hydrogens
                (1, 5, 2), (2, 5, 1),
                (1, 5, 3), (3, 5, 1),
                (1, 5, 4), (4, 5, 1),
                (1, 5, 8), (8, 5, 1),
                (1, 5, 9), (9, 5, 1),
                (1, 5, 10), (10, 5, 1),
                (1, 6, 2), (2, 6, 1)]
    symbols = list("CHHHSOCHHH")
    combinations = list(orutils.generate_atom_combinations(symbols))
    assert len(combinations) == 240
    assert combinations[:22] == expected
