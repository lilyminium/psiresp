import pytest

from psiresp.charge import ChargeConstraintOptions
from psiresp.molecule import Atom


@pytest.fixture()
def methylammonium_nme2ala2_charge_constraints(methylammonium, nme2ala2):
    constraints = ChargeConstraintOptions(symmetric_methyls=False)

    # nmeala2
    nme2ala2_constraints = [(0, [19, 20, 21, 22, 23, 24]),
                            (0.6163, [17]),
                            (-0.5722, [18])]
    for charge, indices in nme2ala2_constraints:
        constraints.add_charge_sum_constraint_for_molecule(nme2ala2,
                                                           charge=charge,
                                                           indices=indices)
    equivalence_constraints = [(nme2ala2, [9, 13]),
                               (nme2ala2, [10, 11, 12, 14, 15, 16]),
                               (methylammonium, [5, 6, 7])]
    for molecule, indices in equivalence_constraints:
        constraints.add_charge_equivalence_constraint_for_molecule(molecule,
                                                                   indices=indices)

    sum_atoms = (Atom.from_molecule(methylammonium, indices=[0, 1, 2, 3])
                 + Atom.from_molecule(nme2ala2, indices=[0, 1, 2, 3, 4, 5, 6, 7]))
    constraints.add_charge_sum_constraint(charge=0, atoms=sum_atoms)
    return constraints
