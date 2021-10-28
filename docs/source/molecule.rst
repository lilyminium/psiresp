Molecules
=========

PsiRESP is built on MolSSI's QC stack. The :class:`psiresp.molecule.Molecule`,
:class:`psiresp.conformer.Conformer`, and :class:`psiresp.orientation.Orientation`
classes each wrap a :class:`qcelemental.models.Molecule` with the :attr:`qcmol`
attribute.

This has several advantages; molecules can be easily hashed, for example, and
they interact natively with the engine-agnostic QCEngine and QCFractal packages.

The hierarchy goes: a :class:`psiresp.molecule.Molecule` contains
:class:`psiresp.conformer.Conformer`s, which contains
:class:`psiresp.orientation.Orientation`s.

----------
Conformers
----------

RESP methods are known to be very conformation-dependent,
so including more conformers increases the likelihood of
getting a charge profile that's more suited to the
conformers explored in simulation.

Adding conformers manually
--------------------------

You can define your own conformers with QCElemental molecules:


.. ipython::

    from psiresp.tests.datafiles import DMSO
    import qcelemental as qcel
    import psiresp as sip

    qcdmso = qcel.models.Molecule.from_file(DMSO, dtype="xyz")
    dmso = sip.Molecule(qcmol=qcdmso)
    # No conformers are generated automatically
    assert dmso.n_conformers == 0
    dmso.add_conformer(qcmol=qcdmso)
    assert dmso.n_conformers == 1
    print(dmso.conformers)

Or directly with coordinates:

.. ipython::

    from psiresp.tests.datafiles import DMSO_O1
    dmso_o1 = qcel.models.Molecule.from_file(DMSO_O1, dtype="xyz")
    dmso.add_conformer_with_coordinates(dmso_o1.geometry, units="bohr")
    assert dmso.n_conformers == 2
    print(dmso.conformers)


Automatically generating conformers
-----------------------------------

However, automatically generating conformers is probably easiest
and likely to get better results. 