Installation
============

Much of the functionality in PsiRESP depends on `RDKit`_, `Psi4`_, and
`QCFractal`_.
These packages are easiest distributed through
`Anaconda`_, so it is highly recommended to install PsiRESP
through ``conda-forge``. You can use normal ``conda`` for this.
However, you may occasionally encounter errors where ``conda`` hangs or times out,
or finds conflicting dependencies. Therefore
the **best way to install PsiRESP is by using mamba**,
a very good drop-in alternative for ``conda``.

You can install ``mamba`` from ``conda-forge`` too::

  conda install -c conda-forge mamba

From there, you can use ``mamba`` to install fully-featured PsiRESP::

  mamba install -c conda-forge psiresp

It is generally recommended to install this fully-featured
version, as typical usage involving Psi4 and SMILES will require
all dependencies. Note -- even fully-featured PsiRESP does not install Psi4
by default, as it is often a good idea to have separate
environments for a PsiRESP job and the actual (QCFractal-managed)
Psi4 computation. To install Psi4::

  mamba install -c psi4 psi4

However, you may have already pre-computed
molecule geometries, grids, and electrostatic potentials.
(An example of using PsiRESP in this minimal way is provided in the example tutorials).
Alternatively, you may only want to install a subset of the above dependencies,
which are quite heavy. Therefore, a version with minimal
dependencies is provided on both conda-forge and PyPI. Via conda,
it can be installed with:

.. code-block:: bash

  mamba install -c conda-forge psiresp-base


Only the minimal version is on PyPI:

.. code-block:: bash

  pip install psiresp


.. note::

    For versions of PsiRESP older than 0.4.1,
    using a plain ``conda`` install without ``mamba``
    may result in very old versions of ``qcfractal``.
    In that case, please add an extra pin when you install::

      conda install -c conda-forge psiresp==0.3.1 "qcfractal>0.15"


--------------------
Building from source
--------------------

In order to build from source, clone the repository:

.. code-block:: bash

  git clone https://github.com/lilyminium/psiresp
  cd psiresp

Create an environment with required dependencies:

.. code-block:: bash

  conda env create -f devtools/conda-envs/environment.yaml
  conda activate psiresp

Then build from source:

.. code-block:: bash

  # build the package
  pip install -e .  # or pip install . if not creating a development environment


To run tests:

.. code-block:: bash

  cd psiresp/tests/
  pytest . --disable-pytest-warnings


------------
Dependencies
------------

The core dependencies of PsiRESP are:

  * `qcelemental <https://docs.qcarchive.molssi.org/projects/QCElemental/en/stable/>`_
  * `numpy <https://numpy.org/>`_
  * `scipy <https://scipy.org/>`_
  * `pydantic <https://pydantic-docs.helpmanual.io/>`_

Additional dependencies for full functionality are:

  * `psi4 <https://psicode.org/>`_
  * `geomeTRIC <https://github.com/leeping/geomeTRIC>`_
  * `qcengine <https://docs.qcarchive.molssi.org/projects/qcengine/en/stable/>`_
  * `QCFractal`_
  * `rdkit <https://www.rdkit.org/>`_


Psi4 and RDKit are only available via ``conda``, so it is best to use ``conda``
to create your Python environment. An environment file is provided for
use with ``conda``, as demonstrated above.


.. _RDKit: https://www.rdkit.org/
.. _Psi4: https://psicode.org/
.. _Anaconda: https://anaconda.org/anaconda/python
.. _QCFractal: https://docs.qcarchive.molssi.org/projects/qcfractal/en/latest/