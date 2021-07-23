Using PsiRESP on HPC
====================

Parallelization
---------------

The quantum chemistry computations in PsiRESP are by far and away the
most computationally expensive parts of PsiRESP. Fortunately, they are
also largely independent of each other and can be run in parallel.


This can be done two ways:

* by running QM jobs in parallel subprocesses during script runtime
* by setting ``execute_qm=False`` and running the jobs in parallel using
  your own job management system. This is frequently a better approach if you
  expect your QM jobs to require significant memory or disk space, and if the
  resources available to your main script do not suffice.

For the former approach, you can provide the number of processes,
number of threads, and memory available to :meth:`psiresp.Resp.run`.
A :class:`concurrent.futures.ProcessPoolExecutor` will be created with
``nprocs`` processes, so that ``nprocs`` Psi4 jobs will be able to run
at once. Each of these Psi4 jobs will have ``nthreads`` threads available
for Psi4 to parallelize its own code over.

For the latter approach and in many cases of the former,
a PsiRESP job will need multiple submissions to complete.
Therefore PsiRESP provides input/output options that write out intermediate files
and read from these files in order to continue a computation. Please see
`io_options`_ for more.


Currently, the only parallelized portion of the code is the Psi4 jobs.
The rest of the process, such as vdW grid generation, is done in serial.
This decision was made under the assumption that the resources and time
required pale in comparison to the QM jobs. Please raise an issue on the
`Issue Tracker`_ if this cost becomes prohibitive.




.. _`Issue Tracker`: https://github.com/lilyminium/psiresp/issues