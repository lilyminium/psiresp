Usage
=====

PsiRESP is built on Psi4 and MolSSI's QC stack. While it is theoretically possible to use
other packages for the QM calculations, PsiRESP is written to use Psi4 as seamlessly as possible.

The general process of computing RESP charges is as follows:

#. Generate some number of conformers
#. (Optional) Use Psi4 to optimize the geometry of each conformer
#. Generate some number of orientations for each conformer
#. Compute the wavefunction of each orientation with Psi4
#. Generate a grid of points around each molecule
#. Evaluate the electrostatic potential (ESP) of the molecule on the grid points
#. (Optional) Set charge constraints
#. Fit charges to these charge constraints and ESPs according to specified RESP options


---------------
Simple examples
---------------

PsiRESP uses a :class:`qcfractal.server.FractalServer` to manage
resources with QM computations. However, it is not always possible
or practical to have a server running in a different process; for
example, if you want to use PsiRESP in a Jupyter notebook, or within
a Python script. Each of the examples shown in this section will use
a :class:`qcfractal.snowflake.FractalSnowflake` temporary server
as an example.

.. ipython:: python

    import qcfractal.interface as ptl
    from psiresp.testing import FractalSnowflake
    server = FractalSnowflake()
    client = ptl.FractalClient(server, verify=False)


If running this in a Jupyter notebook, QCFractal recommends
a :class:`qcfractal.snowflake.FractalSnowflakeHandler`.
    

Calculating charges of one molecule with a temporary server
-----------------------------------------------------------

For our first example, we choose a small molecule (water) at a
low level of theory so that computations finish in a manageable
amount of time. I will use the RDKit 
package to create the molecule from SMILES. 

Now let's create and visualize the molecule.

.. ipython:: python

    from rdkit import Chem
    rd_dmso = Chem.MolFromSmiles("CS(=O)C")
    @savefig dmso.png width=4in
    repr(rd_dmso)


Note that there are no conformers generated
when the PsiRESP molecule is first created.

.. ipython:: python

    import psiresp
    dmso = psiresp.Molecule.from_rdkit(rd_dmso, optimize_geometry=True)
    print(dmso.conformers)


The default RESP options in PsiRESP make for what can be considered
canonical RESP: a 2-stage restrained fit, where hydrogens are
excluded from the restraint, and the scale factors for the asymptote
limits of the hyperbola restraint are 0.0005 and 0.001 for the first
and second stage respectively. The typical method and basis set are
"hg/6-31g*", but we go with "b3lyp/sto-3g" here to save time.

.. ipython:: python
    :okwarning:

    geometry_options = psiresp.QMGeometryOptimizationOptions(
        method="b3lyp", basis="sto-3g")
    esp_options = psiresp.QMEnergyOptions(
        method="b3lyp", basis="sto-3g",
    )
    job = psiresp.Job(molecules=[dmso],
                    qm_optimization_options=geometry_options,
                    qm_esp_options=esp_options,
                    )

    job.run(client=client)
    print(job.charges)




Calculating charges of two molecules with a temporary server
------------------------------------------------------------

We can also calculate the charges of multiple molecules at once.
This is particularly handy for setting charge constraints between molecules,
e.g. constraining groups of atoms in both molecules to sum to 0.

When we set up these molecules, let's turn off geometry optimization
to save some time.


.. ipython:: python

    nme2ala2 = psiresp.Molecule.from_smiles("CC(=O)NC(C)(C)C(NC)=O", optimize_geometry=False)
    methylammonium = psiresp.Molecule.from_smiles("C[NH3+]", optimize_geometry=False)


Let us set up some charge constraints:

.. ipython:: python

    constraints = psiresp.ChargeConstraintOptions()
    nme_smiles = "CC(=O)NC(C)(C)C([N:1]([H:2])[C:3]([H:4])([H:5])([H:6]))=O"
    nme_indices = nme2ala2.get_smarts_matches(nme_smiles)
    print(nme_indices)
    constraints.add_charge_sum_constraint_for_molecule(nme2ala2,
                                                       charge=0,
                                                       indices=nme_indices[0])
    methyl_atoms = methylammonium.get_atoms_from_smarts("C([H])([H])([H])")
    ace_atoms = nme2ala2.get_atoms_from_smarts("C([H])([H])([H])C(=O)N([H])")
    constraint_atoms = methyl_atoms[0] + ace_atoms[0]
    constraints.add_charge_sum_constraint(charge=0, atoms=constraint_atoms)


We can also constrain atoms to have equivalent charges. For example,
the below constrains the hydrogens of the two middle methyls to all
have the same charge.


.. ipython:: python

    h_smiles = "C(C([H:2])([H:2])([H:2]))(C([H:2])([H:2])([H:2]))"
    h_atoms = nme2ala2.get_atoms_from_smarts(h_smiles)[0]
    print(len(h_atoms))
    constraints.add_charge_equivalence_constraint(atoms=h_atoms)


.. ipython:: python
    :okwarning:

    job_multi = psiresp.Job(molecules=[methylammonium, nme2ala2],
                            charge_constraints=constraints,
                            qm_optimization_options=geometry_options,
                            qm_esp_options=esp_options,)
    job_multi.run(client=client)
    print(job_multi.charges[0])
    print(job_multi.molecules[0].to_smiles())
    print(job_multi.charges[1])
    print(job_multi.molecules[1].to_smiles())

----------------------
On a computing cluster
----------------------

The quantum chemistry computations in PsiRESP are by far and away the
most computationally expensive parts of PsiRESP. Fortunately, they are
also largely independent of each other and can be run in parallel.

Using QCFractal
---------------

One way to do this is to use a persistent
:class:`qcfractal.server.FractalServer` rather than a Snowflake version.
On a supercomputer, the process should go:

**1. Submit a cheap, single-core job starting up the QCFractal server**

This can be cheap but should be long-lasting, as the server needs
to manage the job queue. A folder should be given to host the server files.
Below, I give an example of commands to initialize and start a server
for a Slurm job.

.. code-block:: bash

    qcfractal-server init --base-folder "/tmp/${SLURM_JOBID}" --port 7777 --max-active-services 300 --query-limit 100000
    qcfractal-server start --base-folder "/tmp/${SLURM_JOBID}"

**2. Submit jobs for queue managers to compute the tasks**

These are the processes that actually run the computations, so
should include everything necessary -- GPU nodes, multiple cores, etc.
Below is an example of starting a manager in a job that has requested
12 cpus. ``$NODE`` should be the IP address of the node that the server
has been started on in step 1.

.. code-block:: bash

    NODE="hpc3-l18-01"
    qcfractal-manager --verbose --fractal-uri "${NODE}:7777" --verify False --tasks-per-worker 3 --cores-per-worker 4 --memory-per-worker 160 --update-frequency 5


**Submit your Python script**

Within your Python script, you no longer need to create a server;
that has been done in step 1. Instead, the client created in the script
needs the address of the server: ::

    import qcfractal.interface as ptl
    NODE = "hpc3-l18-01"
    PORT = 7777
    client = ptl.FractalClient(f"{NODE}:{PORT}")


Running QM jobs manually
------------------------

However, it may not always be possible to keep a server running;
for example, you may have low walltime limits, or may not be able
to communicate between nodes, or may simply not have the resources
to do all the computation on one machine. In that case, PsiRESP
will write out job inputs for you.

To trigger this behaviour, pass ``client=None`` into the job. ::

    job.run()


