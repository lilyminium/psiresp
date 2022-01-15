import pytest

import qcelemental as qcel

from psiresp.tests.datafiles import (DMSO, METHYLAMMONIUM_OPT,
                                     NME2ALA2_OPT_C1, NME2ALA2_OPT_C2,
                                     ETHANE_JSON,
                                     )


@pytest.fixture
def qcmol(request):
    return qcel.models.Molecule.from_file(request.param, dtype="xyz")


@pytest.fixture
def dmso_qcmol():
    return qcel.models.Molecule.from_file(DMSO, dtype="xyz", molecular_charge=0, molecular_multiplicity=1)


@pytest.fixture
def methylammonium_qcmol():
    return qcel.models.Molecule.from_file(
        METHYLAMMONIUM_OPT,
        dtype="xyz",
        molecular_charge=1,
        molecular_multiplicity=1,
        connectivity=[
            [0, 1, 1],
            [0, 2, 1],
            [0, 3, 1],
            [0, 4, 1],
            [4, 5, 1],
            [4, 6, 1],
            [4, 7, 1],
        ]
    )


@pytest.fixture
def nme2ala2_c1_opt_qcmol():
    return qcel.models.Molecule.from_file(
        NME2ALA2_OPT_C1,
        dtype="xyz",
        molecular_charge=0,
        molecular_multiplicity=1,
        connectivity=[
            [0, 1, 1],
            [0, 2, 1],
            [0, 3, 1],
            [0, 4, 1],
            [4, 5, 2],
            [4, 6, 1],
            [6, 7, 1],
            [6, 8, 1],
            [8, 9, 1],
            [9, 10, 1],
            [9, 11, 1],
            [9, 12, 1],
            [8, 13, 1],
            [13, 14, 1],
            [13, 15, 1],
            [13, 16, 1],
            [8, 17, 1],
            [17, 18, 2],
            [17, 19, 1],
            [19, 20, 1],
            [19, 21, 1],
            [21, 22, 1],
            [21, 23, 1],
            [21, 24, 1],
        ]
    )


@pytest.fixture
def nme2ala2_c2_opt_qcmol():
    return qcel.models.Molecule.from_file(NME2ALA2_OPT_C2, dtype="xyz", molecular_charge=0, molecular_multiplicity=1)


@pytest.fixture
def cc_qcmol():
    return qcel.models.Molecule.parse_file(ETHANE_JSON)
