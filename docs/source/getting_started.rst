Getting Started
===============

------------
Installation
------------

The easiest approach to install the latest release is to use pip::

  # pip
  pip install psiresp


If you need the latest development environment, build from source::

  git clone https://github.com/lilyminium/psiresp
  cd psiresp
  # create new environment with dependencies
  conda env create -f devtools/conda-envs/resp_env.yaml
  conda activate psiresp
  # build the package
  python setup.py install


To run tests::

  cd psiresp/tests/
  pytest . --disable-pytest-warnings


--------------------
Single molecule RESP
--------------------

In general, you can jump straight in with PsiRESP once you have a molecule.
Several example molecule files are provided in the tests. For example, this XYZ file of DMSO::

    10
    ! from R.E.D. examples/2-Dimethylsulfoxide
    C   3.7787218489   0.7099460365  -8.4358800149
    H   3.7199547803   1.7353551063  -8.0912740044
    H   3.3727951491   0.0324988805  -7.6949604622
    H   3.2200309413   0.6050316563  -9.3564160613
    S   5.4843687520   0.2699083657  -8.7873057009
    O   5.4949906162  -1.1820495711  -9.0993845212
    C   6.1255577314   0.4439602615  -7.1184894575
    H   7.1686363815   0.1575024332  -7.1398743610
    H   6.0389920737   1.4725205702  -6.7894907284
    H   5.5889517256  -0.2186737389  -6.4509246884


Resp objects can be constructed directly from this file:

.. ipython::

    import psiresp
    from psiresp.tests.datafiles import DMSO

    resp = psiresp.Resp.from_molfile(DMSO)
    resp.coordinates

Or from Psi4 molecules obtained some other way:

.. ipython::

    import psi4
    with open(DMSO, "r") as f:
        content = f.read()
    mol = psi4.core.Molecule.from_string(content, dtype="xyz")
    resp = psiresp.Resp(mol)
    resp.coordinates

:class:`~psiresp.resp.Resp` accepts a number of optional arguments. These can be broken down
into the categories below, and in fact are managed in the mixin parent classes in parentheses:

* molecule options and conformer generation (:class:`~psiresp.mixins.resp_base.RespMoleculeOptions`) :
  These describe the RESP molecule's charge and multiplicity. Conformer generation options such as
  `max_generated_conformers` and `min_conformer_rmsd` tune the conformers automatically generated
  with RDKit (they do *not* affect user-provided conformers). `minimize_conformer_geometries`
  specifies whether to minimize geometries using a force field -- this is not the same as optimizing
  the geometry using the qm options.
* RESP fitting options (:class:`~psiresp.mixins.resp.RespOptions`) : these control the fitting.
  Options include whether to run a two-stage job, whether the fit is restrained, and the
  scale factors of the hyperbola penalty.
* ESP grid generation options (:class:`~psiresp.mixins.grid.GridOptions`) : these pertain to the
  ESP grid generation -- for example, the vdW radii used to generate the grid, the vdW scale
  factors, and the point density.
* QM job options (:class:`~psiresp.mixins.qm.QMOptions`) : these control the QM jobs used to
  optimize the geometry and compute the electrostatic potential. Options include the
  `qm_method`, `qm_basis_set`, and the `solvent`.
* Input/output options (:class:`~psiresp.mixins.io.IOMixin`) : these options specify whether
  to save the QM output files and intermediate job files, and whether to read the said files in
  where available. Writing and loading can be very helpful when the job must be broken up due
  to walltime limits. The `directory_path` specifies which directory to write or load those files
  from.

For example, to construct a typical two-stage HF/6-31G* RESP in the gas phase::

    resp = psiresp.Resp(mol,
                        stage_2=True, restrained=True, ihfree=True,  # resp options
                        hyp_a1=0.0005, hyp_a2=0.001, hyp_b=0.1,  # resp options
                        use_radii="msk", vdw_point_density=1.0,  # grid options
                        qm_method="hf", qm_basis_set="6-31g*", solvent=None,  # qm options
                        save_output=False, load_input=False,  # io options
                        )


:class:`~psiresp.resp.Resp` accepts a number of other key arguments that provide
base options for creating new conformers and orientations, as well as charge constraints:

* `conformer_options`: (:class:`~psiresp.mixins.conformer.ConformerOptions`) :
  You can ask the 
* `orientation_options`: (:class:`~psiresp.mixins.conformer.OrientationOptions`) : 
  These mostly just control the input/output.
* `charge_constraint_options` (:class:`~psiresp.mixins.charge_constraints.ChargeConstraintOptions`)

The Resp instance does not immediately generate conformers, so you can modify the conformer options
after creating the Resp object. 
