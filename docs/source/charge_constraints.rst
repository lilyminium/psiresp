.. _constraints-label:

Charge constraints
==================

The charge constraints used to fit charges to the ESPs can have
a substantial effect on the resulting charges. You can both
add manual charge constraints yourself (described in the section
below, "Custom charge constraints") and set general options.

The key options are:

* symmetric_methyls (default: True)
* symmetric_methylenes (default: True)
* symmetric_atoms_are_equivalent (default: False)
* split_conformers (default: False)
* constrain_methyl_hydrogens_between_conformers (default: False)

The first three options deal with symmetry.
They constrain methyl hydrogens, methylene hydrogens,
and symmetric atoms to have the same charge, respectively.
Symmetric atoms are determined from the graph representation only,
rather than from 3D coordinates.
It is generally a good idea to have them all as True, although
in some cases where the 3D coordinates differentiate symmetric groups,
you may want to keep `symmetric_atoms_are_equivalent=False`.

`split_conformers` affects how the constraint matrices are constructed.
When `split_conformers=False` (the default), conformers are treated separately, and
equivalence constraints ensure that atoms are given the same charge
across multiple molecules.
When `split_conformers=True`, all conformers are merged into one matrix
so that they are essentially averaged, without equivalence constraints needed.

When conformers are split, `constrain_methyl_hydrogens_between_conformers`
affects whether equivalence constraints are set between conformers for methyl
and methylene hydrogen atoms, for the first stage of a two-stage fit. All atoms
are always given equivalence constraints between conformers during
the last stage of the fit.
Therefore, this option is only relevant if `split_conformers=True`. Using
`split_conformers=True, constrain_methyl_hydrogens_between_conformers=False`
is the approach first proposed by Cornell et al., 1993 in one of the
original RESP papers, where the charges methyl/ene hydrogens are free to
vary in the first stage.

However, using `split_conformers=False` generates charges more in line
with other packages, and for this reason that is the recommended default.
It also saves considerably on memory because the matrices are smaller.



-------------------------
Custom charge constraints
-------------------------

PsiRESP offers two forms of constraints:
a :class:`~psiresp.charge.ChargeSumConstraint` and
a :class:`~psiresp.charge.ChargeEquivalenceConstraint`.

A :class:`~psiresp.charge.ChargeSumConstraint` constrains
one or a group of atoms to a *single* charge. For example, this is very helpful
if you are parametrizing a residue in a large molecule, and you need
the cap atoms to sum to 0.

A :class:`~psiresp.charge.ChargeEquivalenceConstraint`
constrains every atom specified to the *same charge*. For example, you could
manually constrain all hydrogens around a single carbon to the same charge.
For this reason, a :class:`~psiresp.charge.ChargeEquivalenceConstraint`
must contain at least two atoms.

All of these are contained in
:class:`~psiresp.charge.ChargeConstraintOptions`.
ChargeConstraintOptions are passed to a :class:`~psiresp.job.Job`.
As ChargeConstraintOptions may contain charge constraints for molecules
that are not considered in the :class:`~psiresp.job.Job`, the job will
create :class:`~psiresp.charge.MoleculeChargeConstraints` to work on.
Users should typically not interact with
:class:`~psiresp.charge.MoleculeChargeConstraints`
themselves, but let the :class:`~psiresp.job.Job` handle it.
:class:`~psiresp.charge.MoleculeChargeConstraints` contains helper
methods such as adding symmetry constraints for sp3 hydrogens.

Creating a constraint requires specifying the atoms that
the constraint applies to. This can be done in multiple ways.
For example, intra-molecular constraints can be created
from the molecule and atom indices:

.. ipython:: python

    import psiresp
    nme2ala2 = psiresp.Molecule.from_smiles("CC(=O)NC(C)(C)C(NC)=O")
    constraints = psiresp.ChargeConstraintOptions()
    constraints.add_charge_sum_constraint_for_molecule(nme2ala2,
                                                       charge=0,
                                                       indices=[0, 1])
    print(constraints.charge_sum_constraints)

These indices can be obtained through SMARTS matching:

.. ipython:: python

    nme_smiles = "CC(=O)NC(C)(C)C([N:1]([H:2])[C:3]([H:4])([H:5])([H:6]))=O"
    nme_indices = nme2ala2.get_smarts_matches(nme_smiles)
    print(nme_indices)
    constraints.add_charge_equivalence_constraint_for_molecule(nme2ala2,
                                                               indices=nme_indices[0])

Alternatively, you can pass a list of atoms. This is especially useful
for inter-molecular constraints, that involve multiple molecules:

.. ipython:: python

    methylammonium = psiresp.Molecule.from_smiles("C[NH3+]")
    methyl_atoms = methylammonium.get_atoms_from_smarts("C([H])([H])([H])")
    ace_atoms = nme2ala2.get_atoms_from_smarts("C([H])([H])([H])C(=O)N([H])")
    constraint_atoms = methyl_atoms[0] + ace_atoms[0]
    constraints.add_charge_sum_constraint(charge=0, atoms=constraint_atoms)
    constraints.charge_sum_constraints[-1]

You can also indirectly add constraints with the ``symmetric_methylenes``
and ``symmetric_methyls`` terms. These add a :class:`~psiresp.charge.ChargeEquivalenceConstraint`
for the appropriate hydrogens.

.. note::

    For now, detecting sp3 carbons requires accurate chemical perception.
    For reliable symmetry detection, it is highly advisable to create Molecules
    from SMILES, RDKit molecules, or QCElemental molecules with the connectivity
    specified.

While the actual constraints are not created in
:class:`~psiresp.charge.ChargeConstraintOptions`, they are specified in
:class:`~psiresp.charge.MoleculeChargeConstraints`. MoleculeChargeConstraints
are created by a Job; users should not typically create their own or
interact with it. They contain methods for detecting and merging
redundant constraints. For example, we create constraint options
where a constraint for nme2ala2 is added twice,
and a constraint is added that includes atoms from both nme2ala2
and methylammonium:

.. ipython:: python

    constraints = psiresp.ChargeConstraintOptions(symmetric_methyls=True,
                                                  symmetric_methylenes=True)
    # add this constraint twice
    constraints.add_charge_sum_constraint_for_molecule(nme2ala2,
                                                       indices=nme_indices[0])
    constraints.add_charge_sum_constraint_for_molecule(nme2ala2,
                                                       indices=nme_indices[0])
    # add constraint with both nme2ala2 and methylammonium
    constraints.add_charge_sum_constraint(charge=0, atoms=constraint_atoms)
    print(len(constraints.charge_sum_constraints))
    print(len(constraints.charge_equivalence_constraints))

When we create :class:`~psiresp.charge.MoleculeChargeConstraints` with
only the nme2ala2 molecule, the redundant constraint is removed:

.. ipython:: python

    mol_constraints = psiresp.charge.MoleculeChargeConstraints.from_charge_constraints(
                        constraints,
                        molecules=[nme2ala2],
                        )
    print(len(mol_constraints.charge_sum_constraints))


And the sp3 equivalences are added:

.. ipython:: python

    print(len(mol_constraints.charge_equivalence_constraints))
