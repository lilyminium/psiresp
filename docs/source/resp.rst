RESP
====

----------
Background
----------

The electrostatic potential :math:`V` is the potential generated
by the nuclei and electrons in a molecule.
At any point in space :math:`r`, it is:

.. math::
    V(\vec{r}) = \sum_{n=1}^{nuclei} \frac{Z_n}{|\vec{r} - \vec{R}_n|} - \int \frac{\rho(\vec{r’})}{|\vec{r} - \vec{r’}|} d\vec{r}

If we generate a grid of points around the molecule and evaluate the
potential felt at each grid point, we can set up a system of linear
equations to determine the partial charges for the atoms in the
molecule that would best fit the potential. Because the grid of points
is typically generated at certain radii from each atom, forming a 
"surface layer" around the molecule, the system of linear
equations is commonly called "surface constraints" in the code and
throughout the documentation.

You can add your own charge constraints to the "surface constraints"
to form an overall constraint matrix. These charge constraints can
control what charge a group of atoms should sum to 
(:class:`~psiresp.charge.ChargeSumConstraint`) or if one atom
should have an equivalent charge to other atoms
(:class:`~psiresp.charge.ChargeEquivalenceConstraint`).

The equations represented by the constraint matrix
can be solved for the charges of best fit,
as first published by Singh and Kollman in 1984 :cite:p:`singh1984`.

Commonly, a "restrained fit" is performed to derive the final charges :cite:p:`bayly1993,cornell1993,cieplak1995`.

The hyperbolic restraint has the form:

.. math::

    \chi_{penalty} = a\sum_{n=1}^{nuclei} ((q_{n}^{2} + b^2)^{1/2} – b)


:math:`a` defines the asymptotic limits of the penalty, and corresponds to
:attr:`~psiresp.resp.RespOptions.resp_a1` and
:attr:`~psiresp.resp.RespOptions.resp_a2` for the stage 1 and stage 2
fits, respectively.
:math:`b` defines the width of the penalty, and corresponds to
:attr:`~psiresp.resp.RespOptions.resp_b`.

If you only want a one-stage fit, the process stops here.
In a two-stage fit, the typical charge model in AMBER and CHARMM
force fields, the above is repeated. The difference between the
stages is in the charge restraints. In the first stage, all charges
are free to vary. In the second stage fit, atoms without an equivalence
constraint are fixed (i.e. their charges remain static from stage 1).
However, atoms within an equivalence constraint are free to vary.
In more technical detail, this is the process for each stage:

    1. All charge equivalence constraints are ignored,
       and the charges of all atoms are free to vary.
       This includes the hydrogens around sp3 carbons,
       which we would expect to be symmetric or equivalent.
    2. The charge equivalence constraints are added back in the
       second stage. For all atoms that are *not* involved in
       an equivalence constraint, their charges are fixed to
       the stage 1 charges. Now the constraint matrix is only
       solved to calculate the charges of the equivalent atoms.
       

.. note::

    sp3 carbons where the attached hydrogens are involved in a constraint,
    are also not given a fixed charge in stage 2 fits, but left free to vary.




---------------
Practical steps
---------------

The general, practical process of computing RESP charges is as follows:

#. Generate some number of conformers
#. (Optional) Use Psi4 to optimize the geometry of each conformer
#. Generate some number of orientations for each conformer
#. Compute the wavefunction of each orientation with Psi4
#. Generate a grid of points around each molecule
#. Evaluate the electrostatic potential (ESP) of the molecule on the grid points
#. (Optional) Set charge constraints
#. Fit charges to these charge constraints and ESPs according to specified RESP options


All of these are handled for you under the hood with :meth:`psiresp.job.Job.run`.


----------------------
Pre-configured classes
----------------------

The table below gives a broad overview of the pre-configured classes.

.. table::
    :widths: 30 50 20

    +----------------------------------+------------------------------------+-------------------------+
    | Class                            | Description                        | Reference               |
    +==================================+====================================+=========================+
    | :class:`psiresp.configs.RespA1`  | A 2-stage restrained fit           | :cite:t:`bayly1993`,    |
    |                                  | in the gas phase at hf/6-31g*      | :cite:t:`cornell1993`,  |
    |                                  |                                    | :cite:t:`cieplak1995`   |
    +----------------------------------+------------------------------------+-------------------------+
    | :class:`psiresp.configs.RespA2`  | A 1-stage restrained fit           |                         |
    |                                  | in the gas phase at hf/6-31g*      |                         |
    +----------------------------------+------------------------------------+-------------------------+
    | :class:`psiresp.configs.EspA1`   | A 1-stage unrestrained fit         | :cite:t:`singh1984`     |
    |                                  | in the gas phase at hf/6-31g*      |                         |
    +----------------------------------+------------------------------------+-------------------------+
    | :class:`psiresp.configs.EspA2`   | A 1-stage unrestrained fit         |                         |
    |                                  | in the gas phase at hf/sto-3g      |                         |
    +----------------------------------+------------------------------------+-------------------------+
    | :class:`psiresp.configs.ATBResp` | A 2-stage restrained fit in        | :cite:t:`malde2011`     |
    |                                  | implicit water at b3lyp/6-31g*     |                         |
    +----------------------------------+------------------------------------+-------------------------+
    | :class:`psiresp.configs.Resp2`   | A 2-stage restrained fit           | :cite:t:`schauperl2020` |
    |                                  | at pw6b95/aug-cc-pV(D+d)Z,         |                         |
    |                                  | in both vacuum and implicit water. |                         |
    |                                  | Charges are interpolated           |                         |
    |                                  | between the two phases.            |                         |
    +----------------------------------+------------------------------------+-------------------------+
