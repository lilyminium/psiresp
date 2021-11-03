Charge constraints
==================

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