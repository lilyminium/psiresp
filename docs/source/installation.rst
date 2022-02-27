Installation
============

Much of the functionality in PsiRESP depends on `RDKit`_, `Psi4`_, and
`QCFractal`_.
These packages are easiest distributed through
`Anaconda`_, so it is **highly recommended** to install PsiRESP
through ``conda``.

.. code-block:: bash

  conda install -c conda-forge psiresp

The above dependencies can be difficult to install and unnecessary for
PsiRESP's core functionality. Therefore, a version with minimal
dependencies is provided on both conda-forge and PyPI. Via conda,
it can be installed with:

.. code-block:: bash

  conda install -c conda-forge psiresp-base


Only the minimal version is on PyPI:

.. code-block:: bash

  pip install psiresp

It is recommended to install additional dependencies via conda.

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
  python setup.py develop  # or python setup.py install if not creating a development environment


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