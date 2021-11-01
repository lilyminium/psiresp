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
Minimal example
---------------

PsiRESP uses a :class:`qcfractal.server.FractalServer` to manage
resources with QM computations. However, it is not always possible
or practical to have a server running in a different process; for
example, if you want to use PsiRESP in a Jupyter notebook, or within
a Python script. Within a Python script, QCFractal recommends a
:class:`qcfractal.snowflake.FractalSnowflake`; within a Jupyter notebook,
:class:`qcfractal.snowflake.FractalSnowflakeHandler`.

For now, if using a `FractalSnowflake`, it is recommended to use the
patched version in :class:`psiresp.testing.FractalSnowflake`.

.. ipython::

    In [1]: import qcfractal.interface as ptl
    In [2]: from psiresp.testing import FractalSnowflake
    In [3]: import psiresp
    In [4]: server = FractalSnowflake()
    In [5]: client = ptl.FractalClient(server, verify=False)
    In [6]: dmso = psiresp.Molecule.from_smiles("CS(=O)C")
    In [7]: job = psiresp.Job(molecules=[dmso])
    In [8]: job.run(client=client)
    In [9]: print(job.charges)
    Out [9]:
    [array([-0.1419929225688832,  0.174096498208119 , -0.5070885448455941,
            -0.0658571428969831,  0.0992069671540124,  0.0992069671540124,
             0.0992069671540124,  0.0810737368804347,  0.0810737368804347,
             0.0810737368804347])]
    In [10]: print(dmso.to_smiles())
    Out [10]:
    [C:1](-[S:2](=[O:3])-[C:4](-[H:8])(-[H:9])-[H:10])(-[H:5])(-[H:6])-[H:7]



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

    qcfractal-server init --base-folder "/tmp/${SLURM_JOBID}" --port 7777 \
        --max-active-services 300 --query-limit 100000
    qcfractal-server start --base-folder "/tmp/${SLURM_JOBID}"

**2. Submit jobs for queue managers to compute the tasks**

These are the processes that actually run the computations, so
should include everything necessary -- GPU nodes, multiple cores, etc.
Below is an example of starting a manager in a job that has requested
12 cpus. ``$NODE`` should be the IP address of the node that the server
has been started on in step 1.

.. code-block:: bash

    NODE="hpc3-l18-01"
    qcfractal-manager --verbose --fractal-uri "${NODE}:7777" --verify False \
        --tasks-per-worker 3 --cores-per-worker 4 --memory-per-worker 160 \
        --update-frequency 5


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


