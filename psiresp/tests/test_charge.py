import pytest
from numpy.testing import assert_allclose, assert_equal
import numpy as np
import scipy.sparse

import psiresp
from psiresp.charge import (ChargeSumConstraint,
                            ChargeEquivalenceConstraint,
                            ChargeConstraintOptions,
                            )
from psiresp.molecule import Atom
from psiresp.job import Job
from psiresp.constraint import SparseGlobalConstraintMatrix

from psiresp.tests.datafiles import DMSO_STAGE_2_A, DMSO_STAGE_2_B


def test_charge_sum_constraint(dmso):
    constraint = ChargeSumConstraint.from_molecule(dmso, indices=[1, 2])
    assert constraint.atoms == {Atom(molecule=dmso, index=1),
                                Atom(molecule=dmso, index=2)}
    assert constraint.charge == 0
    assert_equal(constraint.molecules, np.array([dmso, dmso]))
    assert constraint.molecule_set == {dmso}

    molinc = {hash(dmso): 4}
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

    molinc = {hash(dmso): 4}
    indices = constraint.get_atom_indices(molecule_increments=molinc)
    assert_allclose(indices, [5, 6, 7])
    row = constraint.to_row_constraint(10, molecule_increments=molinc)
    reference = [[0, 0, 0, 0, 0, -1, 1, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, -1, 1, 0, 0]]
    assert_allclose(row, reference)


def test_options_setup():
    pytest.importorskip("rdkit")

    nme2ala2 = psiresp.Molecule.from_smiles("CC(=O)NC(C)(C)C(NC)=O")
    methylammonium = psiresp.Molecule.from_smiles("C[NH3+]")
    constraints = psiresp.ChargeConstraintOptions()
    nme_smiles = "CC(=O)NC(C)(C)C([N:1]([H:2])[C:3]([H:4])([H:5])([H:6]))=O"
    nme_indices = nme2ala2.get_smarts_matches(nme_smiles)
    constraints.add_charge_sum_constraint_for_molecule(nme2ala2, charge=0,
                                                       indices=nme_indices[0])
    methyl_atoms = methylammonium.get_atoms_from_smarts("C([H])([H])([H])")
    assert len(methyl_atoms) == 1
    assert len(methyl_atoms[0]) == 4
    ace_atoms = nme2ala2.get_atoms_from_smarts("C([H])([H])([H])C(=O)N([H])")
    assert len(ace_atoms) == 1
    assert len(ace_atoms[0]) == 8
    constraint_atoms = methyl_atoms[0] + ace_atoms[0]
    constraints.add_charge_sum_constraint(charge=0, atoms=constraint_atoms)

    h_smiles = "C([C:1]([H:2])([H:2])([H:2]))([C:1]([H:2])([H:2])([H:2]))"
    h_atoms = nme2ala2.get_atoms_from_smarts(h_smiles)[0]
    constraints.add_charge_equivalence_constraint(atoms=h_atoms)

    assert len(constraints.charge_sum_constraints) == 2
    assert len(constraints.charge_equivalence_constraints) == 1

    assert len(constraints.charge_sum_constraints[0]) == 6
    assert len(constraints.charge_sum_constraints[1]) == 12


class TestMoleculeChargeConstraints:

    # @pytest.mark.slow
    def test_add_constraints_from_charges(self, dmso, fractal_client):
        pytest.importorskip("psi4")

        charge_options = ChargeConstraintOptions(symmetric_methyls=True,
                                                 symmetric_methylenes=True)
        job = Job(molecules=[dmso],
                  charge_constraints=charge_options
                  )
        job.generate_conformers()
        job.generate_orientations()
        job.compute_orientation_energies(client=fractal_client)
        job.compute_esps()

        assert len(charge_options.charge_sum_constraints) == 0
        assert len(charge_options.charge_equivalence_constraints) == 0

        constraints = job.generate_molecule_charge_constraints()
        assert len(constraints.charge_sum_constraints) == 0
        assert len(constraints.charge_equivalence_constraints) == 2

        constraints.add_constraints_from_charges(
            [-0.43877469, 0.14814998, 0.17996033, 0.18716814, 0.35743529,
             -0.5085439, -0.46067469, 0.19091725, 0.15500465, 0.18935764]
        )
        assert len(constraints.charge_sum_constraints) == 2
        assert len(constraints.charge_equivalence_constraints) == 2

        surface_constraints = job.construct_surface_constraint_matrix()
        matrix = SparseGlobalConstraintMatrix.from_constraints(surface_constraints,
                                                               constraints)

        ref_a = np.loadtxt(DMSO_STAGE_2_A)
        ref_b = np.loadtxt(DMSO_STAGE_2_B)

        assert_allclose(matrix.coefficient_matrix.toarray(), ref_a)
        assert_allclose(matrix.constant_vector, ref_b)
