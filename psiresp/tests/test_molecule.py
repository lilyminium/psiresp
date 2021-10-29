import pytest

from psiresp.molecule import Molecule
from psiresp.conformer import ConformerGenerationOptions


def test_molecule_defaults(dmso_qcmol):
    mol = Molecule(qcmol=dmso_qcmol)
    mol.generate_orientations()
    assert len(mol.conformers) == 1
    assert len(mol.conformers[0].orientations) == 1


def test_conformer_generation(nme2ala2_c1_opt_qcmol):
    options = ConformerGenerationOptions(n_max_conformers=5)
    mol = Molecule(qcmol=nme2ala2_c1_opt_qcmol, conformer_generation_options=options)
    mol.generate_conformers()
    mol.generate_orientations()
    assert len(mol.conformers) == 1
    assert len(mol.conformers[0].orientations) == 1
