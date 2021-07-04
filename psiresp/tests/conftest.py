import pytest

from .base import coordinates_from_xyzfile, psi4mol_from_xyzfile
from .datafiles import (DMSO, DMSO_O1, DMSO_O2, DMSO_O3, DMSO_O4,
                        NME2ALA2_C1,
                        )


@pytest.fixture()
def dmso_coordinates():
    return coordinates_from_xyzfile(DMSO)


@pytest.fixture()
def dmso_psi4mol():
    return psi4mol_from_xyzfile(DMSO)


@pytest.fixture()
def dmso_o1_coordinates():
    return coordinates_from_xyzfile(DMSO_O1)


@pytest.fixture()
def dmso_o1_psi4mol():
    return psi4mol_from_xyzfile(DMSO_O1)


@pytest.fixture()
def dmso_o2_coordinates():
    return coordinates_from_xyzfile(DMSO_O2)


@pytest.fixture()
def dmso_o2_psi4mol():
    return psi4mol_from_xyzfile(DMSO_O2)


@pytest.fixture()
def dmso_o3_psi4mol():
    return psi4mol_from_xyzfile(DMSO_O3)


@pytest.fixture()
def dmso_o4_psi4mol():
    return psi4mol_from_xyzfile(DMSO_O4)


@pytest.fixture()
def dmso_orientation_psi4mols(dmso_o1_psi4mol, dmso_o2_psi4mol,
                              dmso_o3_psi4mol, dmso_o4_psi4mol):
    return [dmso_o1_psi4mol, dmso_o2_psi4mol,
            dmso_o3_psi4mol, dmso_o4_psi4mol]


@pytest.fixture()
def nme2ala2_c1_psi4mol():
    return psi4mol_from_xyzfile(NME2ALA2_C1)
