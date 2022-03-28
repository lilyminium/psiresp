# PsiRESP
😪-RESP

| **Latest release** | [![Last release tag](https://img.shields.io/github/release-pre/lilyminium/psiresp.svg)](https://github.com/lilyminium/psiresp/releases) ![GitHub commits since latest release (by date) for a branch](https://img.shields.io/github/commits-since/lilyminium/psiresp/latest)  [![Documentation Status](https://readthedocs.org/projects/psiresp/badge/?version=latest)](https://psiresp.readthedocs.io/en/latest/?badge=latest)|
| :------ | :------- |
| **Installation** | [![Conda (channel only)](https://img.shields.io/conda/vn/conda-forge/psiresp)]((https://anaconda.org/conda-forge/psiresp)) ![Conda Downloads](https://img.shields.io/conda/dn/conda-forge/psiresp.svg) [![PyPI version](https://badge.fury.io/py/psiresp.svg)](https://pypi.org/project/psiresp/) ![PyPI - Downloads](https://img.shields.io/pypi/dm/psiresp) |
| **Status** | [![GH Actions Status](https://github.com/lilyminium/psiresp/actions/workflows/gh-ci.yaml/badge.svg)](https://github.com/lilyminium/psiresp/actions?query=branch%3Amaster+workflow%3Agh-ci) [![codecov](https://codecov.io/gh/lilyminium/psiresp/branch/master/graph/badge.svg)](https://codecov.io/gh/lilyminium/psiresp/branch/master) [![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/lilyminium/psiresp.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/lilyminium/psiresp/context:python) |
| **Community** | [![License: LGPL v3](https://img.shields.io/badge/License-LGPL_v3-blue.svg)](https://www.gnu.org/licenses/lgpl-3.0) |

PsiRESP is a package for calculating atomic partial charges from
restrained and unrestrained electrostatic potential fits using Psi4.
It is highly flexible, configurable, easy to use, and totally written in Python.
It supports fitting to multiple orientations and conformers,
as well as both intra-molecular and inter-molecular charge constraints for
multi-molecule fits.
It is written to correspond closely with existing tools such as the
[RESP ESP charge Derive (R.E.D.)](https://upjv.q4md-forcefieldtools.org/RED/) tools.

As of now, the following implementations are well-tested to reproduce results from existing tools, primarily R.E.D.:

* standard 2-stage RESP (convenience class: psiresp.configs.TwoStageRESP)
* standard 1-stage RESP (convenience class: psiresp.configs.OneStageRESP)
* standard unrestrained ESP (convenience class: psiresp.configs.ESP)

These implementations are not as well-tested:
* ESP using HF/STO-3G (convenience class: psiresp.configs.WeinerESP) -- Psi4 seems to minimize to a relatively different geometry than GAMESS with STO-3G.
* psiresp.configs.ATBRESP, mimicking the method used by the [Automated Topology Builder](https://atb.uq.edu.au/) is not tested at all. The published methods do not indicate the point density, moreover, the results generated seem to have changed since the original paper. **Use at your own risk.**
* psiresp.configs.RESP2, as the methods are expensive

### Installation

The recommended way to install PsiRESP is via [anaconda](https://anaconda.org/anaconda/python),
as the required dependencies are most easily installed distributed through ``conda``.

For the fully featured version, install:

```
conda install -c conda-forge -c psi4 psiresp psi4
```

This will pull in all dependencies necessary for full functionality, including
[RDKit](https://www.rdkit.org/), [Psi4](https://psicode.org/) and
[QCFractal](https://docs.qcarchive.molssi.org/projects/qcfractal/en/latest/).

For minimal functionality, install:

```
conda install -c conda-forge psiresp-base
```

`psiresp-base` installs the package with minimal dependencies, so that
only functionality that does not depend on RDKit, Psi4,or QCFractal is available.

The library can also be installed with minimal dependencies via Pypi:

```
pip install psiresp
```


Alternatively, to build from source: 

* clone this repository
* create a new environment with dependencies
* build the package

```
git clone https://github.com/lilyminium/psiresp.git
cd psiresp
conda env create -f devtools/conda-envs/environment.yaml
conda activate psiresp
pip install .
```

Please see [the Installation docs](https://psiresp.readthedocs.io/en/latest/installation.html) for more information on installation and dependencies.

### Example

Examples for PsiRESP are provided as tutorials both [online](https://psiresp.readthedocs.io/en/latest/examples/README.html)
and as downloadable Jupyter notebooks in the
[examples folder](https://github.com/lilyminium/psiresp/tree/review-updates/docs/source/examples).
More information can also be found in the [documentation](https://psiresp.readthedocs.io/en/latest/).

A minimal example is provided below, running a standard 2-stage restrained electrostatic potential fit (Bayly et al., 1993).
This requires the full installation of `psiresp`, instead of the minimal `psiresp-base`,
as it makes use of RDKit, Psi4 and QCFractal.

```python
import psiresp
from psiresp.testing import FractalSnowflake
import qcfractal.interface as ptl

# set up server and client
server = FractalSnowflake()
client = ptl.FractalClient(server)

# set up molecule
dmso = psiresp.Molecule.from_smiles("CS(=O)C")

# set up job
job = psiresp.Job(molecules=[dmso])
charges = job.run(client=client)
```

### Contributing

All contributions are welcomed! This can include sharing bug reports, bug fixes, requesting or adding new features, or improving the documentation.
If you notice any issues or have feature requests, please open an issue on the [Issue tracker](https://github.com/lilyminium/psiresp/issues).
Otherwise, please check out the [Contributing](https://psiresp.readthedocs.io/en/latest/contributing.html) page in the documentation.


### Copyright

Copyright (c) 2020, Lily Wang

#### Acknowledgements

Project based on the
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.2.

Pre-configured models and reorientation algorithm are written to directly match results from 
[RESP ESP charge Derive (R.E.D.)](https://upjv.q4md-forcefieldtools.org/RED/).
Dupradeau, F.-Y. et al. The R.E.D. tools: advances in RESP and ESP charge derivation and force field library building. Phys. Chem. Chem. Phys. 12, 7821 (2010).

ATBRESP tries to match results from [Automated Topology Builder (A.T.B.)](https://atb.uq.edu.au/).
Malde, A. K. et al. An Automated Force Field Topology Builder (ATB) and Repository: Version 1.0. J. Chem. Theory Comput. 7, 4026–4037 (2011).

RESP2 tries to match results from [RESP2](https://github.com/MSchauperl/RESP2).
Schauperl, M. et al. Non-bonded force field model with advanced restrained electrostatic potential charges (RESP2). Commun Chem 3, 1–11 (2020).

Some tests compare results to output from [resp](https://github.com/cdsgroup/resp), the current RESP plugin 
for Psi4. 
Alenaizan, A., Burns, L. A. & Sherrill, C. D. Python implementation of the restrained electrostatic potential charge model. International Journal of Quantum Chemistry 120, e26035 (2020).
