import pytest
from numpy.testing import assert_almost_equal
import numpy as np

from psiresp.utils import psi4utils, ANGSTROM_TO_BOHR

from ..base import (coordinates_from_xyzfile,
                    psi4mol_from_xyzfile,
                    assert_coordinates_almost_equal,
                    get_angstrom_coordinates,
                    )
from ..datafiles import DMSO


@pytest.mark.parametrize("loader", [
    coordinates_from_xyzfile,
    psi4mol_from_xyzfile,
])
def test_as_coordinates(loader, dmso_coordinates):
    mol_or_coords = loader(DMSO)
    coords = psi4utils.as_coordinates(mol_or_coords)
    assert isinstance(coords, np.ndarray)
    assert_coordinates_almost_equal(coords, dmso_coordinates)


@pytest.mark.parametrize("name_in, name_out", [
    ("name", "name"),
    (None, "default")
])
def test_psi4mol_with_coordinates(dmso_psi4mol, dmso_o1_coordinates,
                                  name_in, name_out):
    psi4mol = psi4utils.psi4mol_with_coordinates(dmso_psi4mol,
                                                 dmso_o1_coordinates,
                                                 name=name_in)
    geometry = get_angstrom_coordinates(psi4mol)
    assert_coordinates_almost_equal(geometry, dmso_o1_coordinates)
    assert psi4mol.name() == name_out


def test_get_mol_spec(dmso_psi4mol):
    # psi4 handles most of this -- just check the prefix and suffix
    mol_spec = psi4utils.get_mol_spec(dmso_psi4mol)
    lines = [line for line in mol_spec.split("\n") if line]
    assert lines[0].strip() == "molecule default {"
    assert lines[1].strip() == "units Angstrom"
    assert lines[2].strip() == "0 1"
    assert lines[-1].strip() == "}"


def test_set_psi4mol_geometry(dmso_psi4mol, dmso_o1_coordinates):
    charge = dmso_psi4mol.molecular_charge()
    multiplicity = dmso_psi4mol.multiplicity()
    psi4utils.set_psi4mol_coordinates(dmso_psi4mol, dmso_o1_coordinates)
    assert dmso_psi4mol.molecular_charge() == charge
    assert dmso_psi4mol.multiplicity() == multiplicity
    geometry = get_angstrom_coordinates(dmso_psi4mol)
    assert_coordinates_almost_equal(geometry, dmso_o1_coordinates)


@pytest.mark.parametrize("file, increment, ref_sp3_ids", [
    (DMSO, 0, {1: [2, 3, 4], 7: [8, 9, 10]}),
    (DMSO, 10, {11: [12, 13, 14], 17: [18, 19, 20]}),
])
def test_get_sp3_ch_ids(file, increment, ref_sp3_ids):
    psi4mol = psi4mol_from_xyzfile(file)
    sp3_ids = psi4utils.get_sp3_ch_ids(psi4mol, increment=increment)
    assert sp3_ids == ref_sp3_ids
