# PsiRESP
😪-RESP

| **Latest release** | [![PyPI version](https://badge.fury.io/py/psiresp.svg)](https://badge.fury.io/py/psiresp) [![Documentation Status](https://readthedocs.org/projects/psiresp/badge/?version=latest)](https://psiresp.readthedocs.io/en/latest/?badge=latest)|
| :------ | :------- |
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

As of now, the following implementations are well-tested:

* standard 2-stage RESP (convenience class: psiresp.configs.RespA1)
* standard 1-stage RESP (convenience class: psiresp.configs.RespA2)
* standard unrestrained ESP (convenience class: psiresp.configs.EspA1)

These implementations are not as well-tested:
* ESP using HF/STO-3G (convenience class: psiresp.configs.EspA2) -- Psi4 seems to minimize to a relatively different geometry than GAMESS with STO-3G.
* psiresp.configs.ATBResp, mimicking the method used by the [Automated Topology Builder](https://atb.uq.edu.au/) is not tested at all. The published methods do not indicate the point density, moreover, the results generated seem to have changed since the original paper. **Use at your own risk.**
* psiresp.configs.Resp2, as the methods are expensive

### Installation

Create a new conda environment with dependencies. In general, installing the dependencies required will be difficult without `conda`, as both Psi4 and RDKit are most easily distributed through conda.

```
conda env create -f devtools/conda-envs/environment.yaml
conda activate psiresp
```

To build from source, clone this repository and install the package.

```
git clone https://github.com/lilyminium/psiresp.git
cd psiresp
python setup.py install
```

Please see [the Installation docs](https://psiresp.readthedocs.io/en/latest/installation.html) for more information on installation and dependencies.

### Example

For example, running a standard 2-stage restrained electrostatic potential fit (Bayly et al., 1993) as standard in AMBER 
(implemented as RESP-A1 in R.E.D.):

```python
   import psiresp
   from psiresp.testing import FractalSnowflake  # if using Jupyter, use FractalSnowflakeHandler below
   # from qcfractal import FractalSnowflakeHandler
   import qcfractal.interface as ptl

   # set up server and client
   server = FractalSnowflake()
   client = ptl.FractalClient(server)

   # set up conformer generation options
   conformer_options = psiresp.ConformerGenerationOptions(n_max_conformers=2)  # generate at most 2 conformers
   
   # set up molecule
   dmso = psiresp.Molecule.from_smiles("CS(=O)C", charge=0, multiplicity=1,
                                       optimize_geometry=True,  # optimize conformers
                                       conformer_generation_options=conformer_options
                                       )
      
   # set up charge constraints
   charge_constraints = psiresp.ChargeConstraintOptions(
      symmetric_methyls=True,  # make methyl Hs around carbons the same charge
      symmetric_methylenes=True,  # make methylene Hs around carbons the same charge
   )
   # constrain S and O atoms to sum to -0.19617
   so_atoms = dmso.get_atoms_from_smarts("S=O")[0]
   charge_constraints.add_charge_sum_constraint(charge=-0.19617, atoms=so_atoms)
   # constrain two C atoms to have the same charge
   cc_atoms = dmso.get_atoms_from_smarts("[C:1]S(=O)[C:2]")[0]
   charge_constraints.add_charge_equivalence_constraint(atoms=cc_atoms)
   
   # set up job
   job = psiresp.Job(
      molecules=[dmso],
      resp_options=psiresp.RespOptions(
         stage_2=True,  # run 2-stage RESP
         resp_a1=0.0005,  # hyperbola restraints for stage 1
         resp_a2=0.001,  # hyperbola restraints for stage 2
         restrained_fit=True,  # restrain ESP fit
      ),
      qm_optimization_options=psiresp.QMGeometryOptimizationOptions(
         basis="6-31g*",
         method="hf",
      ),
      qm_esp_options=psiresp.QMEnergyOptions(
         basis="6-31g*",
         method="hf",
      ),
      charge_constraints=charge_constraints,
   )

   charges = job.run(client=client)

```
Alternatively, use the preconfigured RespA1 class in ``psiresp.configs``.
This sets up the `grid_options`, `resp_options`,
`qm_optimization_options`, and `qm_esp_options`

```python

   job = psiresp.RespA1(molecules=[dmso])
   # constrain S and O atoms to sum to -0.19617
   so_atoms = dmso.get_atoms_from_smarts("S=O")[0]
   job.charge_constraints.add_charge_sum_constraint(charge=-0.19617, atoms=so_atoms)
   # constrain two C atoms to have the same charge
   cc_atoms = dmso.get_atoms_from_smarts("[C:1]S(=O)[C:2]")[0]
   job.charge_constraints.add_charge_equivalence_constraint(atoms=cc_atoms)

   charges = job.run()
```

Please see the [examples](https://psiresp.readthedocs.io/en/latest/examples/README.html) and [documentation](https://psiresp.readthedocs.io/en/latest/) for more.

### Copyright

Copyright (c) 2020, Lily Wang

#### Acknowledgements

Project based on the
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.2.

Pre-configured models and reorientation algorithm are written to directly match results from 
[RESP ESP charge Derive (R.E.D.)](https://upjv.q4md-forcefieldtools.org/RED/).
Dupradeau, F.-Y. et al. The R.E.D. tools: advances in RESP and ESP charge derivation and force field library building. Phys. Chem. Chem. Phys. 12, 7821 (2010).

ATBResp tries to match results from [Automated Topology Builder (A.T.B.)](https://atb.uq.edu.au/).
Malde, A. K. et al. An Automated Force Field Topology Builder (ATB) and Repository: Version 1.0. J. Chem. Theory Comput. 7, 4026–4037 (2011).

Resp2 tries to match results from [RESP2](https://github.com/MSchauperl/RESP2).
Schauperl, M. et al. Non-bonded force field model with advanced restrained electrostatic potential charges (RESP2). Commun Chem 3, 1–11 (2020).

Some tests compare results to output from [resp](https://github.com/cdsgroup/resp), the current RESP plugin 
for Psi4. 
Alenaizan, A., Burns, L. A. & Sherrill, C. D. Python implementation of the restrained electrostatic potential charge model. International Journal of Quantum Chemistry 120, e26035 (2020).
