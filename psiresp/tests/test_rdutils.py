import pytest

import qcelemental as qcel
import numpy as np
from numpy.testing import assert_allclose

pytest.importorskip("rdkit")


@pytest.mark.parametrize("guess, n_bonds", [
    (True, 9),
    (False, 0)
])
def test_rdmol_from_qcelemental_guess_connectivity(dmso_qcmol, guess, n_bonds):
    from psiresp.rdutils import rdmol_from_qcelemental
    rdmol = rdmol_from_qcelemental(dmso_qcmol, guess_connectivity=guess)
    rdatoms = [a.GetSymbol() for a in rdmol.GetAtoms()]
    assert rdatoms == list("CHHHSOCHHH")
    assert rdmol.GetNumConformers() == 1
    assert rdmol.GetNumBonds() == n_bonds
    if n_bonds:
        assert rdmol.GetBondBetweenAtoms(4, 5).GetBondTypeAsDouble() == 2.0

    rdgeom = np.array(rdmol.GetConformer(0).GetPositions())
    BOHR_TO_ANGSTROM = qcel.constants.conversion_factor("bohr", "angstrom")
    qcgeom = dmso_qcmol.geometry * BOHR_TO_ANGSTROM

    assert_allclose(rdgeom, qcgeom, atol=5e-05)


def test_rdmol_from_qcmol_methylammonium(methylammonium_qcmol):
    from psiresp.rdutils import rdmol_from_qcelemental
    rdmol = rdmol_from_qcelemental(methylammonium_qcmol, guess_connectivity=True)
    rdatoms = [a.GetSymbol() for a in rdmol.GetAtoms()]
    assert rdatoms == list("CHHHNHHH")
    n = rdmol.GetAtomWithIdx(4)
    assert n.GetSymbol() == "N"
    assert n.GetFormalCharge() == 1
