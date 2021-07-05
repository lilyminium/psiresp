import pytest

from psiresp import mixins


def test_create_moleculemixin(dmso_psi4mol):
    obj = mixins.MoleculeMixin(psi4mol=dmso_psi4mol)
    assert obj.psi4mol is dmso_psi4mol
    assert obj.name == "default"
    assert obj.n_atoms == 10
    # assert obj.charge == 0
    # assert obj.multiplicity == 1
