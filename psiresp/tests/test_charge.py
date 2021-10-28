import pytest
from numpy.testing import assert_allclose, assert_equal
import numpy as np
import scipy.sparse

import psiresp
from psiresp.charge import (ChargeSumConstraint,
                            ChargeEquivalenceConstraint,
                            BaseChargeConstraintOptions,
                            ChargeConstraintOptions,
                            MoleculeChargeConstraints
                            )
from psiresp.molecule import Atom
from psiresp.job import Job

from psiresp.tests.datafiles import DMSO_STAGE_2_A, DMSO_STAGE_2_B


def test_charge_sum_constraint(dmso):
    constraint = ChargeSumConstraint.from_molecule(dmso, indices=[1, 2])
    assert constraint.atoms == {Atom(molecule=dmso, index=1),
                                Atom(molecule=dmso, index=2)}
    assert constraint.charge == 0
    assert_equal(constraint.molecules, np.array([dmso, dmso]))
    assert constraint.molecule_set == {dmso}

    molinc = {dmso: 4}
    indices = constraint.get_atom_indices(molecule_increments=molinc)
    assert_allclose(indices, [5, 6])
    row = constraint.to_row_constraint(10, molecule_increments=molinc)
    assert_allclose(row, [[0, 0, 0, 0, 0, 1, 1, 0, 0, 0]])


def test_charge_equivalence_constraint(dmso):
    constraint = ChargeEquivalenceConstraint.from_molecule(dmso, indices=[1, 2, 3])
    assert constraint.atoms == {Atom(molecule=dmso, index=1),
                                Atom(molecule=dmso, index=2),
                                Atom(molecule=dmso, index=3)}
    assert_equal(constraint.molecules, np.array([dmso, dmso, dmso]))
    assert constraint.molecule_set == {dmso}

    molinc = {dmso: 4}
    indices = constraint.get_atom_indices(molecule_increments=molinc)
    assert_allclose(indices, [5, 6, 7])
    row = constraint.to_row_constraint(10, molecule_increments=molinc)
    reference = [[0, 0, 0, 0, 0, -1, 1, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, -1, 1, 0, 0]]
    assert_allclose(row, reference)


# class TestBaseChargeConstraintOptions:

#     @pytest.fixture(scope="function")
#     def options(self):
#         constraint = BaseChargeConstraintOptions(

#         )

#     def test_unite_overlapping_equivalences(self)

class TestMoleculeChargeConstraints:
    def test_add_constraints_from_charges(self, dmso, dmso_qcmol, fractal_client):
        charge_options = ChargeConstraintOptions(symmetric_methyls=True,
                                                 symmetric_methylenes=True)
        job = Job(molecules=[dmso],
                  charge_constraints=charge_options
                  )
        job.generate_conformers()
        job.generate_orientations()
        job.compute_orientation_energies(client=fractal_client)
        # job.compute_esps()

        # assert len(charge_options.charge_sum_constraints) == 0
        # assert len(charge_options.charge_equivalence_constraints) == 0

        # constraints = job.generate_molecule_charge_constraints()
        # assert len(constraints.charge_sum_constraints) == 0
        # assert len(constraints.charge_equivalence_constraints) == 2

        # constraints.add_constraints_from_charges(
        #     [-0.43877469, 0.14814998, 0.17996033, 0.18716814, 0.35743529,
        #      -0.5085439, -0.46067469, 0.19091725, 0.15500465, 0.18935764]
        # )
        # assert len(constraints.charge_sum_constraints) == 2
        # assert len(constraints.charge_equivalence_constraints) == 2

        # surface_constraints = job.construct_molecule_constraint_matrix()
        # matrix = SparseGlobalConstraintMatrix.from_constraints(surface_constraints,
        #                                                        constraints)

        # ref_a = np.loadtxt(DMSO_STAGE_2_A)
        # ref_b = np.loadtxt(DMSO_STAGE_2_B)

        # assert_allclose(matrix.a.toarray(), ref_a)
        # assert_allclose(matrix.b, ref_b)
