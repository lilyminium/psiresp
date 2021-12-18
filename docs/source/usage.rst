Usage
=====

PsiRESP is built on Psi4 and MolSSI's QC stack. While it is theoretically possible to use
other packages for the QM calculations, PsiRESP is written to use Psi4 as seamlessly as possible.

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

---------------
Minimal example
---------------

This is a minimal example for demonstration purposes.
Please see the examples in :ref:`examples-label` for
more detailed tutorials.

Let us first create a molecule. This can be done from
an RDKit molecule, QCElemental molecule, or simply from
a SMILES string.

.. code-block:: ipython

    In [1]: import psiresp
    In [2]: dmso = psiresp.Molecule.from_smiles("CS(=O)C")

Charge computation is always carried out with a
:class:`psiresp.job.Job`. A default job will calculate
charges using common RESP settings, i.e. a two-stage
fit in the gas phase at hf/6-31g*. However,
**by default, only one conformer and orientation is used** -- this
is to prevent overwriting any user-provided conformers.
For the sake of this minimal example, we will keep that setting.
However, it is highly recommended to use multiple conformers
and multiple orientations; please see :ref:`resp-label`_ for more
information.

.. code-block:: ipython

    In [3]: job = psiresp.Job(molecules=[dmso])

Next, we need to figure out how to calculate the
quantum chemistry jobs. 
PsiRESP uses a :class:`qcfractal.server.FractalServer` to manage
resources with QM computations. However, it is not always possible
or practical to have a server running in a different process; for
example, if you want to use PsiRESP in a Jupyter notebook, or within
a Python script. Within a Python script, QCFractal recommends a
:class:`qcfractal.snowflake.FractalSnowflake`; within a Jupyter notebook,
:class:`qcfractal.snowflake.FractalSnowflakeHandler`.

Alternatively, you may not want to use a server at all, but to run the
QM computations yourselves. In that case, pass ``client=None``.
Please see :ref:`manual_qm` for more information.


.. note::
    For now, if using a `FractalSnowflake`, it is recommended to use the
    patched version in :class:`psiresp.testing.FractalSnowflake`.

The code below creates a QCFractal server and client.

.. code-block:: ipython

    In [4]: import qcfractal.interface as ptl
    In [5]: from psiresp.testing import FractalSnowflake
    In [6]: server = FractalSnowflake()
    In [7]: client = ptl.FractalClient(server, verify=False)

We can then run the job by passing it the client. It will
use this client to submit jobs to, and retrieve jobs from,
the server.

.. code-block:: ipython

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



-----------------------------------
Customising RESP charge computation
-----------------------------------

Each of the aspects of computing RESP charges can be customised to correspond
to the implementations used by :cite:t:`bayly1993`, :cite:t:`singh1984`,
:cite:t:`malde2011`, :cite:t:`schauperl2020`, and so on. These require setting options
for grid generation, the QM computation, and the hyperbolic restraints themselves;
please see :ref:`option_classes` for the specific options.

However, for ease of use, PsiRESP also provides pre-configured classes.
A full list is available at :ref:`preconfigured_classes`
as well as :ref:`Pre-configured classes`. In order to use these,
simply replace `Job` with the particular chosen configuration:

.. ipython:: python

    import psiresp
    dmso = psiresp.Molecule.from_smiles("CS(=O)C")
    esp_a1 = psiresp.EspA1(molecules=[dmso])
    print(esp_a1.resp_options)

And use :meth:`~psiresp.configs.EspA1.run()` to run the job, as usual.