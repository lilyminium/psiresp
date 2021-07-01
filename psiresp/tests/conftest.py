import pytest

from .base import coordinates_from_xyzfile, psi4mol_from_xyzfile
from .datafiles import DMSO, DMSO_O1


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
