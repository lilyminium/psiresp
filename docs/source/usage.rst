Usage
=====

PsiRESP is built on Psi4 and MolSSI's QC stack. While it is theoretically possible to use
other packages for the QM calculations, PsiRESP is written to use Psi4 as seamlessly as possible.


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

Alternatively, you may not want to use a server at all, but to run the
QM computations yourselves. In that case, pass ``client=None``.
Please see :ref:`manual_qm` for more information.

For now, if using a `FractalSnowflake`, it is recommended to use the
patched version in :class:`psiresp.testing.FractalSnowflake`.

.. code-block:: ipython

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
    esp_a1 = psiresp.ESP(molecules=[dmso])
    print(esp_a1.resp_options)

And use :meth:`~psiresp.configs.ESP.run()` to run the job, as usual.