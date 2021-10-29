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
a :class:`qcfractal.server.FractalSnowflakeHandler` temporary server
as an example.

.. ipython:: python

    import qcfractal.interface as ptl
    from qcfractal import FractalSnowflakeHandler, FractalSnowflake
    from psiresp.testing import FractalSnowflake



Calculating charges of one molecule with a temporary server
-----------------------------------------------------------

For our first example, we choose a small molecule (water) at a
low level of theory so that computations finish in a manageable
amount of time. I will use the OpenForceField openff-toolkit
package to create the molecule from SMILES. This is not a
dependency of PsiRESP, so you may need to install it with:

.. code-block:: bash

    conda install -c conda-forge openff-toolkit


Now let's create and visualize the molecule.

.. ipython:: python

    from openff.toolkit.topology import Molecule as OFFMolecule
    off_dmso = OFFMolecule.from_smiles("O")
    off_dmso.visualize(backend="rdkit")


The OpenFF toolkit has a handy command to convert to a
QCElemental molecule. Note that there are no conformers generated
when the PsiRESP molecule is first created.

.. ipython:: python

    import psiresp
    off_dmso.generate_conformers(n_conformers=1)
    qc_dmso = off_dmso.to_qcschema()
    dmso = psiresp.Molecule(qcmol=qc_dmso, optimize_geometry=False)
    print(dmso.conformers)


The default RESP options in PsiRESP make for what can be considered
canonical RESP: a 2-stage restrained fit, where hydrogens are
excluded from the restraint, and the scale factors for the asymptote
limits of the hyperbola restraint are 0.0005 and 0.001 for the first
and second stage respectively. The typical method and basis set are
"HF/6-31G*", but we go with "b3lyp/sto-3g" here to save time.

Below, we create a temporary server and client for this job, and then
compute charges with the default options.

.. ipython:: python

    geometry_options = psiresp.QMGeometryOptimizationOptions(
        method="b3lyp", basis="sto-3g",
        g_convergence="nwchem_loose",
        max_iter=1,
        full_hess_every=0,
    )
    esp_options = psiresp.QMEnergyOptions(
        method="b3lyp", basis="sto-3g",
    )
    job = psiresp.Job(molecules=[dmso],
                    qm_optimization_options=geometry_options,
                    qm_esp_options=esp_options,
                    )
    with FractalSnowflakeHandler(ncores=4) as server:
        client = ptl.FractalClient(server)
        print(client)
        job.run(client=client)
        print(job.charges)
        server.stop()



----------------------
On a computing cluster
----------------------

The quantum chemistry computations in PsiRESP are by far and away the
most computationally expensive parts of PsiRESP. Fortunately, they are
also largely independent of each other and can be run in parallel.
