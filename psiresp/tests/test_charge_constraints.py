import pytest

from psiresp.charge_constraints import (AtomId,
                                        ChargeConstraint,
                                        ChargeEquivalence)


def test_atom_increment():
    atom_id = AtomId((1, 1))  # first atom of first molecule
    assert atom_id.absolute_atom_index == 0
    atom_id.atom_increment = 10
    assert atom_id.absolute_atom_index == 10
