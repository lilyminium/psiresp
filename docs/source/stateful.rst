Using PsiRESP on HPC
====================

Parallelization
---------------

The quantum chemistry computations in PsiRESP are by far and away the
most computationally expensive parts of PsiRESP. Fortunately, they are
also largely independent of each other and can be run in parallel.

This can be done two ways:

* by providing a :class:`concurrent.futures.Executor` for submitting QM jobs
  in parallel, during script runtime
* by setting ``execute_qm=False`` and running the jobs in parallel using
  your own job management system. This is frequently a better approach if you
  expect your QM jobs to require significant memory or disk space, and if the
  resources available to your main script do not suffice.






Running over multiple jobs
--------------------------

Job walltime limits are an unfortunate reality of HPC facilities.
In many cases a PsiRESP job will need multiple submissions to complete.
Therefore PsiRESP provides IO options that write out intermediate files
and read from these files in order to continue a computation.




