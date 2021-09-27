import pytest
import MDAnalysis as mda

from psiresp import mixins


@pytest.fixture()
def molecule(dmso_psi4mol):
    return mixins.MoleculeMixin(psi4mol=dmso_psi4mol)


def test_create_moleculemixin(molecule, dmso_psi4mol):
    assert molecule.psi4mol is dmso_psi4mol
    assert molecule.name == "default"
    assert molecule.n_atoms == 10


def test_to_mda(molecule):
    u = molecule.to_mda()
    assert isinstance(u, mda.Universe)
    assert len(u.atoms) == 10


def test_write(tmpdir, molecule):
    with tmpdir.as_cwd():
        molecule.write("test.xyz")
        u2 = mda.Universe("test.xyz")
        assert len(u2.atoms) == 10


def test_clone(molecule):
    new = molecule.clone()
    assert new.name == "default_copy"
    assert new.psi4mol is not molecule.psi4mol
    assert new.n_atoms == 10


def test_clone_with_name(molecule):
    new = molecule.clone("meepmorp")
    assert new.name == "meepmorp"
    assert new.psi4mol is not molecule.psi4mol
    assert new.n_atoms == 10


def test_validate_psi4mol(dmso_psi4mol):
    string = dmso_psi4mol.to_string(dtype="psi4")
    obj = mixins.MoleculeMixin(psi4mol=string)
    assert obj.n_atoms == 10
