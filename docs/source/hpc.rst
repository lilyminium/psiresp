Using PsiRESP on HPC
====================

Parallelization
---------------

The quantum chemistry computations in PsiRESP are by far and away the
most computationally expensive parts of PsiRESP. Fortunately, they are
also largely independent of each other and can be run in parallel.




Currently, the only parallelized portion of the code is the Psi4 jobs.
The rest of the process, such as vdW grid generation, is done in serial.
This decision was made under the assumption that the resources and time
required pale in comparison to the QM jobs. Please raise an issue on the
`Issue Tracker`_ if this cost becomes prohibitive.




.. _`Issue Tracker`: https://github.com/lilyminium/psiresp/issues