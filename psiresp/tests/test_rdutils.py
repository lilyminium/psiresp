import pytest

from rdkit import Chem


def test_rdmol_from_qcelemental(dmso_qcmol):
    rdmol = rdmol_from_qcelemental(dmso_qcmol)
    rdatoms = [a.GetSymbol() for a in rdmol.GetAtoms()]
    assert rdatoms == list("CHHHSOCHHH")
    assert rdmol.GetNumConformers() == 1
    assert rdmol.GetNumBonds() == 9
    assert rdmol.GetBondBetweenAtoms(4, 5).GetBondTypeAsDouble() == 2
