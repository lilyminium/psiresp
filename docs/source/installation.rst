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
  python setup.py install


To run tests::

  cd psiresp/tests/
  pytest . --disable-pytest-warnings


