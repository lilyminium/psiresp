import pytest
from numpy.testing import assert_equal

import psiresp
from psiresp.molecule import Molecule
from psiresp.conformer import ConformerGenerationOptions


def test_molecule_defaults(dmso_qcmol):
    mol = psiresp.Molecule(qcmol=dmso_qcmol)
    mol.generate_orientations()
    assert len(mol.conformers) == 1
    assert len(mol.conformers[0].orientations) == 1


def test_conformer_generation(nme2ala2_c1_opt_qcmol):
    options = ConformerGenerationOptions(n_max_conformers=5)
    mol = psiresp.Molecule(qcmol=nme2ala2_c1_opt_qcmol, conformer_generation_options=options)
    mol.generate_conformers()
    mol.generate_orientations()
    assert len(mol.conformers) == 6
    assert len(mol.conformers[0].orientations) == 1


def test_smarts_searching():
    nme2ala2 = psiresp.Molecule.from_smiles("CC(=O)NC(C)(C)C(NC)=O")
    nme_smiles = "CC(=O)NC(C)(C)C([N:1]([H:2])[C:3]([H:4])([H:5])([H:6]))=O"
    nme_indices = nme2ala2.get_smarts_matches(nme_smiles)
    assert len(nme_indices) == 1
    assert len(nme_indices[0]) == 6
    assert_equal(nme2ala2.qcmol.symbols[list(nme_indices[0])], ["N", "H", "C", "H", "H", "H"])


def test_smarts_unlabeled():
    methylammonium = psiresp.Molecule.from_smiles("C[NH3+]")
    methyl_atoms = methylammonium.get_atoms_from_smarts("C([H])([H])([H])")
    assert len(methyl_atoms) == 1
    assert len(methyl_atoms[0]) == 4
    symbols = [at.symbol for at in methyl_atoms[0]]
    assert_equal(symbols, ["C", "H", "H", "H"])


@pytest.mark.parametrize("insmiles, outsmiles", [
    ("C[NH3+]", "[C:1](-[N+:2](-[H:6])(-[H:7])-[H:8])(-[H:3])(-[H:4])-[H:5]"),
    ("CC(=O)NC(C)(C)C(NC)=O", ("[C:1](-[C:2](=[O:3])-[N:4](-[C:5]"
                               "(-[C:6](-[H:16])(-[H:17])-[H:18])"
                               "(-[C:7](-[H:19])(-[H:20])-[H:21])-"
                               "[C:8](-[N:9](-[C:10](-[H:23])(-[H:24])"
                               "-[H:25])-[H:22])=[O:11])-[H:15])(-[H:12])"
                               "(-[H:13])-[H:14]"))
])
def to_smiles(insmiles, outsmiles):
    mol = psiresp.Molecule.from_smiles(insmiles)
    assert mol.to_smiles() == outsmiles
