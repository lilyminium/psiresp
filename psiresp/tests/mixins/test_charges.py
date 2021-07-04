import pytest
from numpy.testing import assert_almost_equal
import numpy as np

import psiresp
from psiresp.mixins import ChargeConstraintOptions
from psiresp.utils import psi4utils
from ..datafiles import DMSO_STAGE_2_A


def test_get_constraint_matrix_stage_2(dmso_psi4mol):
    options = ChargeConstraintOptions()
    sp3_ids = psi4utils.get_sp3_ch_ids(dmso_psi4mol)
    options.add_sp3_equivalences(sp3_ids)
    charges = [-0.31436137, 0.11376814, 0.14389421, 0.15583091, 0.30951557,
               -0.50568553, -0.33670336, 0.15982101, 0.12029157, 0.15362883]
    options.add_stage_2_constraints(np.array(charges))

    resp = psiresp.Resp(psi4mol=dmso_psi4mol)
    resp.generate_conformers()
    resp.generate_orientations()
    a_matrix = resp.get_a_matrix()
    b_matrix = resp.get_b_matrix()
    constraint_a, _ = options.get_constraint_matrix(a_matrix, b_matrix)
    reference = np.loadtxt(DMSO_STAGE_2_A)

    assert_almost_equal(constraint_a, reference, decimal=5)
