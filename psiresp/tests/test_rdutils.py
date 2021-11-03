import pytest

import qcelemental as qcel
import numpy as np
from numpy.testing import assert_allclose

from psiresp.rdutils import rdmol_from_qcelemental


@pytest.mark.parametrize("guess, n_bonds", [
    (True, 9),
    (False, 0)
])
def test_rdmol_from_qcelemental_guess_connectivity(dmso_qcmol, guess, n_bonds):
    rdmol = rdmol_from_qcelemental(dmso_qcmol, guess_connectivity=guess)
    rdatoms = [a.GetSymbol() for a in rdmol.GetAtoms()]
    assert rdatoms == list("CHHHSOCHHH")
    assert rdmol.GetNumConformers() == 1
    assert rdmol.GetNumBonds() == n_bonds

    rdgeom = np.array(rdmol.GetConformer(0).GetPositions())
    BOHR_TO_ANGSTROM = qcel.constants.conversion_factor("bohr", "angstrom")
    qcgeom = dmso_qcmol.geometry * BOHR_TO_ANGSTROM

    assert_allclose(rdgeom, qcgeom, atol=5e-05)
