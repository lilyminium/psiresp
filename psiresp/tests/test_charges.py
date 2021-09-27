import pytest
from numpy.testing import assert_allclose
import numpy as np

import psiresp
from psiresp import ChargeConstraintOptions, MoleculeChargeConstraints
from psiresp import psi4utils
from .datafiles import DMSO_STAGE_2_A, DMSO_STAGE_2_B


class TestMoleculeChargeConstraints:
    def test_add_constraints_from_charges(self, dmso, tmp_client):
        options = ChargeConstraintOptions(symmetric_methyls=True,
                                          symmetric_methylenes=True)
        constraints = MoleculeChargeConstraints.from_charge_constraints(options,
                                                                        molecules=[dmso])
        constraints.add_constraints_from_charges(
            [-0.43877469, 0.14814998, 0.17996033, 0.18716814, 0.35743529,
             -0.5085439, -0.46067469, 0.19091725, 0.15500465, 0.18935764]
        )
        matrix = constraints.construct_constraint_matrix()

        ref_a = np.loadtxt(DMSO_STAGE_2_A)
        ref_b = np.loadtxt(DMSO_STAGE_2_B)
        assert_allclose(matrix.a.toarray(), ref_a)
        assert_allclose(matrix.b.toarray(), ref_b)
