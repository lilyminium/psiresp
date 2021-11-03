import pytest

from numpy.testing import assert_allclose
from psiresp.moleculebase import generate_atom_combinations, BaseMolecule


def test_generate_atom_combinations():
    symbols = ["H", "C", "C", "O", "N"]
    combinations = generate_atom_combinations(symbols)
    reference = [(1, 2, 3), (3, 2, 1),
                 (1, 2, 4), (4, 2, 1),
                 (1, 3, 4), (4, 3, 1),
                 (2, 3, 4), (4, 3, 2),
                 (1, 2, 0), (0, 2, 1),
                 (1, 3, 0), (0, 3, 1),
                 (1, 4, 0), (0, 4, 1),
                 (2, 3, 0), (0, 3, 2),
                 (2, 4, 0), (0, 4, 2),
                 (3, 4, 0), (0, 4, 3)]

    assert list(combinations) == reference


def test_basemolecule(dmso_qcmol):
    mol = BaseMolecule(qcmol=dmso_qcmol)
    assert mol.n_atoms == 10
    assert mol.coordinates.shape == (10, 3)

    combinations = mol.generate_atom_combinations(n_combinations=2)
    assert combinations == [(0, 4, 5), (5, 4, 0)]

    mol2 = BaseMolecule(qcmol=dmso_qcmol)
    assert hash(mol) == hash(mol2)
