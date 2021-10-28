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
from ..datafiles import (DMSO_ORIENTATION_COORDINATES,
                         DMSO, DMSO_PDB, DMSO_GRO)


def test_get_conformer_coordinates(dmso_rdmol):
    coordinates = rdutils.get_conformer_coordinates(dmso_rdmol)
    ref = np.load(DMSO_ORIENTATION_COORDINATES)
    assert_almost_equal(coordinates, ref, decimal=5)


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