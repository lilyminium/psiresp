Molecules
=========

PsiRESP is built on MolSSI's QC stack. The :class:`psiresp.molecule.Molecule`,
:class:`psiresp.conformer.Conformer`, and :class:`psiresp.orientation.Orientation`
classes each wrap a :class:`qcelemental.models.Molecule` with the :attr:`qcmol`
attribute.

This has several advantages; molecules can be easily hashed, for example, and
they interact natively with the engine-agnostic QCEngine and QCFractal packages.

There is a hierarchy to these molecules.
A :class:`psiresp.molecule.Molecule` contains
:class:`psiresp.conformer.Conformer` s, which contains
:class:`psiresp.orientation.Orientation` s.

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


.. ipython:: python

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

.. ipython:: python

    from psiresp.tests.datafiles import DMSO_O1
    dmso_o1 = qcel.models.Molecule.from_file(DMSO_O1, dtype="xyz")
    dmso.add_conformer_with_coordinates(dmso_o1.geometry, units="bohr")
    assert dmso.n_conformers == 2
    print(dmso.conformers)



Automatically generating conformers
-----------------------------------

However, automatically generating conformers is probably easiest
and likely to get better results. The conformers generated depend
on the :class:`psiresp.conformer.ConformerGenerationOptions`
passed to a :class:`psiresp.molecule.Molecule`.

The process of generating and selecting conformers is as follows:

#. Use RDKit to generate
   :attr:`~psiresp.conformer.ConformerGenerationOptions.n_conformer_pool`
   initial conformers at least
   :attr:`~psiresp.conformer.ConformerGenerationOptions.rms_tolerance`
   angstrom apart in RMSD
#. Keep only the conformers within a certain energy window in kcal/mol.
   This means only those conformers within
   :attr:`~psiresp.conformer.ConformerGenerationOptions.energy_window`
   kcal/mol of the lowest energy conformer are considered for the next step.
#. Select a set with, at most,
   :attr:`~psiresp.conformer.ConformerGenerationOptions.n_max_conformers`
   maximally diverse conformers from the remaining pool.
   Diversity is calculated by heavy atom RMSD.

It is recommmended to geometry optimize these conformers before
generating Orientations from them. :meth:`psiresp.job.Job.run` will
do this automatically, providing
`psiresp.molecule.Molecule.optimize_geometry = True`.


------------
Orientations
------------

It is also recommended to include multiple orientations
for each conformer in the RESP calculation.
The orientations are controlled by the
:attr:`psiresp.molecule.Molecule.reorientations`,
:attr:`psiresp.molecule.Molecule.rotations`, and
:attr:`psiresp.molecule.Molecule.translations` attributes, as well as
:attr:`psiresp.molecule.Molecule.keep_original_orientation`.

:attr:`psiresp.molecule.Molecule.reorientations` and
:attr:`psiresp.molecule.Molecule.rotations` are lists of atom indices, whereas
:attr:`psiresp.molecule.Molecule.translations` is a translation vector.


Reorientations
--------------

Three atom indices must be specified. The first atom becomes the new origin;
the second defines the x-axis from the origin; and the third defines the xy plane.


Rotations
---------

Three atom indices must be specified.
The first two atoms define a vector parallel to the x-axis, while the third defines
a plane parallel to the xy-plane.


Translations
------------
Three floats must be given, as the translation in the x, y, and z axes.


Automatically generating transformations
----------------------------------------

As with Conformers, Orientation specifications can be automatically generated with
:meth:`psiresp.molecule.Molecule.generate_transformations`.

.. note::

    This method does *not* generate the Orientations themselves, but rather
    fills the :attr:`~psiresp.molecule.Molecule.reorientations`,
    :attr:`~psiresp.molecule.Molecule.rotations`, and
    :attr:`~psiresp.molecule.Molecule.translations` lists. This means that
    you can, and should, generate the transformations before generating
    conformers.


If given a desired number of reorientations or rotations, combinations of atoms
will be generated to reorient the molecule around. The method first combines
heavy atoms, before including hydrogens.

If given a desired number of translations, random translation vectors will
be generated between -5 to 5 angstrom on each axis.
