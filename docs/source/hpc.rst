Using PsiRESP on HPC
====================


The quantum chemistry computations in PsiRESP are by far and away the
most computationally expensive parts of PsiRESP. Fortunately, they are
also largely independent of each other and can be run in parallel.

Currently, the only parallelized portion of the code is the Psi4 jobs.
The rest of the process, such as vdW grid generation, is done in serial.
This decision was made under the assumption that the resources and time
required pale in comparison to the QM jobs. Please raise an issue on the
`Issue Tracker`_ if this cost becomes prohibitive.

Using a persistent server with QCFractal
----------------------------------------

One way to do this is to use a persistent
:class:`qcfractal.server.FractalServer`.
If using multiple jobs, the process should go:

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


If running everything on one job with all resources, follow these steps
but just run them as parallel bash scripts or processes.


Using a temporary server with QCFractal
---------------------------------------

Alternatively, if this is not possible, use a
:class:`qcfractal.snowflake.FractalSnowflake` server. Please see
QCFractal's documentation for more, but you are able to specify
``max_workers`` for the ``ProcessPoolExecutor`` to spin up.


Running QM jobs manually
------------------------

Finally, neither of these two options may be suitable. For example,
you may have low walltime limits or other resources, or may not be able
to communicate between nodes, or may simply not have the resources
to do all the computation on one machine. In that case, PsiRESP
will write out job inputs for you.

To trigger this behaviour, simply pass ``client=None`` into the job. ::

    job.run()


This will write Psi4 job files to be run manually and then quit Python.
The output files will be in QCSchema format, with the ``msgpack`` extension;
they are typically run like so:

.. code-block:: bash

    psi4 --qcschema CH6N_795ee2a77c6a4347fd76ed1ab0f8c486d24a2238_c9ce731306cb83b137c5cfd5f69a120483b61005.msgpack


All filenames and commands will be written in ``run_optimization.sh`` or
``run_single_point.sh``, in the directory with the msgpack input files.

Currently, Psi4 does not create a separate output file but writes the
results back into the input file for QCSchema inputs. In order to
continue the job after running Psi4 yourself, call ``job.run()`` again.
This will go through the files and check for successful execution or errors.
If all files have been successfully executed, the job will continue; if
errors are found, an error will be raised with all the found error messages;
and if any files remain to be executed, the job will quit Python again.

For an example job with two different molecules, the directory structure
will look like so by the time the job is complete:

.. code-block::

    .
    └── psiresp_working_directory
        ├── optimization
        │   ├── C7H14N2O2_9ee96ceb2aec1b0d4b5c53ad3ae9e61d546f6717_c9ce731306cb83b137c5cfd5f69a120483b61005.msgpack
        │   ├── C7H14N2O2_abb25794aba793b7bf575666eaefae61736f254e_c9ce731306cb83b137c5cfd5f69a120483b61005.msgpack
        │   ├── CH6N_795ee2a77c6a4347fd76ed1ab0f8c486d24a2238_c9ce731306cb83b137c5cfd5f69a120483b61005.msgpack
        │   └── run_optimization.sh
        └── single_point
            ├── C7H14N2O2_0c7e913c4a8462f7181fc25edf913be91be8d7c4_e746222796fc2c4c5a1f896fa1cc1cefffe7044c.msgpack
            ├── C7H14N2O2_3ec849bebba8d4b8054b40a889b681d861afc28b_e746222796fc2c4c5a1f896fa1cc1cefffe7044c.msgpack
            ├── C7H14N2O2_4d7331ba8d65f69230c53e612f3bc259271675a0_e746222796fc2c4c5a1f896fa1cc1cefffe7044c.msgpack
            ├── C7H14N2O2_5536bc6a52953c07b31bf82f85c8e90f2142cccf_e746222796fc2c4c5a1f896fa1cc1cefffe7044c.msgpack
            ├── C7H14N2O2_5899f84c638c2be4f9e3ba4cd9beedff56c6cc3c_e746222796fc2c4c5a1f896fa1cc1cefffe7044c.msgpack
            ├── C7H14N2O2_607de52ef03791820ad946978d48703837b9338c_e746222796fc2c4c5a1f896fa1cc1cefffe7044c.msgpack
            ├── C7H14N2O2_da2a0eaa440175fbcb099bd6b74de7ac980c9b50_e746222796fc2c4c5a1f896fa1cc1cefffe7044c.msgpack
            ├── C7H14N2O2_daaa75b70b8b2b50a22d1441439a1affcb9be48d_e746222796fc2c4c5a1f896fa1cc1cefffe7044c.msgpack
            ├── CH6N_5522973281a29d00873575078945db705e9e0167_e746222796fc2c4c5a1f896fa1cc1cefffe7044c.msgpack
            ├── CH6N_ecb8b34e0b75ae8e0a73bc53bdd6a4d5c1a5b5c2_e746222796fc2c4c5a1f896fa1cc1cefffe7044c.msgpack
            └── run_single_point.sh


It is important not to change the directory structure,
as the job will look there for files. The filenames should
also not be changed; they are formatted ``{name}_{molecular_hash}_{qm_hash}``,
where the ``name`` is either the name assigned to the QCElemental molecule
or its molecular formula; ``molecular_hash`` is the deterministic
and geometry-dependent `QCElemental hash`_ ; and ``qm_hash`` is the
deterministic hash of the QM options used for the calculation. 

.. _`QCElemental hash`: https://docs.qcarchive.molssi.org/projects/QCElemental/en/stable/model_molecule.html#molecular-hash
.. _`Issue Tracker`: https://github.com/lilyminium/psiresp/issues