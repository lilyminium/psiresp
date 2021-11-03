import pytest

import qcelemental as qcel

from psiresp.tests.datafiles import (DMSO, METHYLAMMONIUM_OPT,
                                     NME2ALA2_OPT_C1, NME2ALA2_OPT_C2)


@pytest.fixture
def qcmol(request):
    return qcel.models.Molecule.from_file(request.param, dtype="xyz")


@pytest.fixture
def dmso_qcmol():
    return qcel.models.Molecule.from_file(DMSO, dtype="xyz", molecular_charge=0, molecular_multiplicity=1)


@pytest.fixture
def methylammonium_qcmol():
    return qcel.models.Molecule.from_file(METHYLAMMONIUM_OPT, dtype="xyz", molecular_charge=1, molecular_multiplicity=1)


@pytest.fixture
def nme2ala2_c1_opt_qcmol():
    return qcel.models.Molecule.from_file(NME2ALA2_OPT_C1, dtype="xyz", molecular_charge=0, molecular_multiplicity=1)


@pytest.fixture
def nme2ala2_c2_opt_qcmol():
    return qcel.models.Molecule.from_file(NME2ALA2_OPT_C2, dtype="xyz", molecular_charge=0, molecular_multiplicity=1)
