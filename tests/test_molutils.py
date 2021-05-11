import pytest

from psiresp import molutils

from .datafiles import ETOH_PDB, ETOH_MOL2, ETOH_XYZ


@pytest.mark.parametrize("filename", ["CCO", ETOH_PDB, ETOH_MOL2])
def test_load_rdmol(filename):
    rdmol = molutils.load_rdmol(filename)
    assert rdmol.GetNumAtoms() == 9


def test_load_rdmol_error():
    err = "Could not parse"
    with pytest.raises(ValueError, match=err):
        rdmol = molutils.load_rdmol(ETOH_XYZ)


@pytest.mark.parametrize("filename",
                         ["CCO", ETOH_PDB, ETOH_MOL2, ETOH_XYZ])
def test_rdmol_from_file(filename):
    rdmol = molutils.rdmol_from_file(filename)
    assert rdmol.GetNumAtoms() == 9


@pytest.mark.parametrize("filename", [ETOH_PDB, ETOH_MOL2, ETOH_XYZ])
def test_psi4mol_from_file(filename):
    psi4mol = molutils.psi4mol_from_file(filename)
    assert psi4mol.natom() == 9


