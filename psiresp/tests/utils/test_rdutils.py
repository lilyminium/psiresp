import pytest
import numpy as np
from numpy.testing import assert_almost_equal
from rdkit import Chem

from psiresp.utils import rdutils, ANGSTROM_TO_BOHR

from ..base import (coordinates_from_xyzfile,
                    psi4mol_from_xyzfile,
                    assert_coordinates_almost_equal,
                    get_angstrom_coordinates,
                    )
from ..datafiles import (DMSO_TPL, DMSO_ORIENTATION_COORDINATES,
                         DMSO, DMSO_PDB, DMSO_GRO)


@pytest.fixture()
def dmso_rdmol():
    return Chem.MolFromTPLFile(DMSO_TPL)


def test_rdmol_to_psi4mols(dmso_rdmol, dmso_orientation_psi4mols):
    psi4mols = rdutils.rdmol_to_psi4mols(dmso_rdmol, name="dmso")
    assert len(psi4mols) == 4
    for i, (actual, ref) in enumerate(zip(psi4mols, dmso_orientation_psi4mols), 1):
        assert_coordinates_almost_equal(actual.geometry().np, ref.geometry().np, decimal=4)
        assert actual.name() == f"dmso_c00{i}"
        assert actual.molecular_charge() == 0
        assert actual.multiplicity() == 1


def test_get_conformer_coordinates(dmso_rdmol):
    coordinates = rdutils.get_conformer_coordinates(dmso_rdmol)
    ref = np.load(DMSO_ORIENTATION_COORDINATES)
    assert_almost_equal(coordinates, ref, decimal=5)


def test_rdmol_from_psi4mol(dmso_psi4mol, dmso_coordinates):
    rdmol = rdutils.rdmol_from_psi4mol(dmso_psi4mol)
    assert rdmol.GetNumAtoms() == 10
    assert rdmol.GetNumConformers() == 1
    conf = rdmol.GetConformer(0)
    coordinates = np.asarray(conf.GetPositions())
    assert_coordinates_almost_equal(coordinates, dmso_coordinates)


def test_add_conformer_from_coordinates(dmso_rdmol, dmso_coordinates):
    assert dmso_rdmol.GetNumConformers() == 4
    rdutils.add_conformer_from_coordinates(dmso_rdmol, dmso_coordinates)
    assert dmso_rdmol.GetNumConformers() == 5
    conf = dmso_rdmol.GetConformer(4)
    coordinates = np.asarray(conf.GetPositions())
    assert_almost_equal(coordinates, dmso_coordinates, decimal=5)


@pytest.mark.parametrize("file", [DMSO, DMSO_PDB, DMSO_GRO])
def test_rdmol_from_file(file, dmso_coordinates):
    rdmol = rdutils.rdmol_from_file_or_string(file)
    coordinates = np.asarray(rdmol.GetConformer(0).GetPositions())
    assert_almost_equal(coordinates, dmso_coordinates, decimal=2)
