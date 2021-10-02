import pytest

import qcelemental as qcel

from psiresp.tests.datafiles import (DMSO, METHYLAMMONIUM_OPT,
                                     NME2ALA2_OPT_C1, NME2ALA2_OPT_C2)


@pytest.fixture
def qcmol(request):
    return qcel.models.Molecule.from_file(request.param)


@pytest.fixture
def dmso_qcmol():
    return qcel.models.Molecule.from_file(DMSO)


@pytest.fixture
def methylammonium_qcmol():
    return qcel.models.Molecule.from_file(METHYLAMMONIUM_OPT)


@pytest.fixture
def nme2ala2_c1_opt_qcmol():
    return qcel.models.Molecule.from_file(NME2ALA2_OPT_C1)


@pytest.fixture
def nme2ala2_c2_opt_qcmol():
    return qcel.models.Molecule.from_file(NME2ALA2_OPT_C2)
