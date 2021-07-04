import pytest
from numpy.testing import assert_almost_equal
import numpy as np

import psiresp
from psiresp.mixins import ChargeConstraintOptions
from psiresp.utils import psi4utils
from ..datafiles import DMSO_STAGE_2_A, DMSO_STAGE_2_B


def test_get_constraint_matrix_stage_2(dmso_psi4mol):
    options = ChargeConstraintOptions()
    sp3_ids = psi4utils.get_sp3_ch_ids(dmso_psi4mol)
    options.add_sp3_equivalences(sp3_ids)
    charges = [-0.43877469, 0.14814998, 0.17996033, 0.18716814, 0.35743529,
               -0.5085439, -0.46067469, 0.19091725, 0.15500465, 0.18935764]
    options.add_stage_2_constraints(np.array(charges))

    resp = psiresp.Resp(psi4mol=dmso_psi4mol)
    resp.generate_conformers()
    resp.generate_orientations()
    a_matrix = resp.get_a_matrix()
    b_matrix = resp.get_b_matrix()
    constraint_a, constraint_b = options.get_constraint_matrix(a_matrix, b_matrix)
    ref_a = np.loadtxt(DMSO_STAGE_2_A)
    ref_b = np.loadtxt(DMSO_STAGE_2_B)

    assert_almost_equal(constraint_a, ref_a, decimal=5)
    assert_almost_equal(constraint_b, ref_b, decimal=5)
