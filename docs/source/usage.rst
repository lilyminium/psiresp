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



Simple examples
---------------



On a computing cluster
----------------------

The quantum chemistry computations in PsiRESP are by far and away the
most computationally expensive parts of PsiRESP. Fortunately, they are
also largely independent of each other and can be run in parallel.
