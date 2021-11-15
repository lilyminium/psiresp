Installation
============

The easiest approach to install the latest release is to use pip::

  # pip
  pip install psiresp


If you need the latest development environment, build from source::

  git clone https://github.com/lilyminium/psiresp
  cd psiresp
  # create new environment with dependencies
  conda env create -f devtools/conda-envs/environment.yaml
  conda activate psiresp
  # build the package
  python setup.py develop  # or python setup.py install if not creating a development environment


To run tests::

  cd psiresp/tests/
  pytest . --disable-pytest-warnings


------------
Dependencies
------------

The core dependencies of PsiRESP are:

  * `psi4 <https://psicode.org/>`_
  * `geomeTRIC <https://github.com/leeping/geomeTRIC>`_
  * `qcelemental <https://docs.qcarchive.molssi.org/projects/QCElemental/en/stable/>`_
  * `qcengine <https://docs.qcarchive.molssi.org/projects/qcengine/en/stable/>`_
  * `qcfractal <https://docs.qcarchive.molssi.org/projects/qcfractal/en/stable/>`_
  * `rdkit <https://www.rdkit.org/>`_
  * `numpy <https://numpy.org/>`_
  * `scipy <https://scipy.org/>`_
  * `pydantic <https://pydantic-docs.helpmanual.io/>`_

Psi4 and RDKit are only available via ``conda``, so it is best to use ``conda``
to create your Python environment. An environment file is provided for
use with ``conda``, as demonstrated above.