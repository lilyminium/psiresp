import pytest

from psiresp.molecule import Molecule


def test_molecule_defaults(dmso_qcmol):
    mol = Molecule(qcmol=dmso_qcmol)
    mol.generate_orientations()
    assert len(mol.conformers) == 1
    assert len(mol.conformers[0].orientations) == 1
