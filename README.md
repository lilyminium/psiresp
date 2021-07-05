# PsiRESP
😪-RESP

[//]: # "Badges"

[![Travis Build Status](https://travis-ci.com/lilyminium/psiresp.svg?branch=master)](https://travis-ci.com/lilyminium/psiresp)
[![codecov](https://codecov.io/gh/lilyminium/psiresp/branch/master/graph/badge.svg)](https://codecov.io/gh/lilyminium/psiresp/branch/master)
[![Documentation Status](https://readthedocs.org/projects/psiresp/badge/?version=latest)](https://psiresp.readthedocs.io/en/latest/?badge=latest)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/lilyminium/psiresp.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/lilyminium/psiresp/context:python)
[![PyPI version](https://badge.fury.io/py/psiresp.svg)](https://badge.fury.io/py/psiresp)
      

A RESP plugin for Psi4.

**Things that probably work:**

* standard 2-stage RESP, gas phase HF/6-31G* with MSK radii (convenience class: psiresp.Multi/RespA1)
* intermolecular charge constraints (use the MultiResp classes for this)
* intramolecular charge constraints (use the Resp classes for this)

**Things that maybe work?**

* 1-stage ESP, B3LYP/6-31G* in implicit solvent with MSK radii. Unknown point density? (convenience class: psiresp.Multi/ATBResp)
* RESP2: PW6B95/aug-cc-pV(D+d)Z in vacuum and solvent with Bondi radii, density=2.5 (convenience class: psiresp.Multi/Resp2)

### Installation

The package is on Pypi, so you can install with `pip`. Note that you will need to install `psi4` separately with `conda`.

```
conda install -c psi4 psi4
pip install psiresp
```

To build from source, clone this repository:

```
git clone https://github.com/lilyminium/psiresp.git
cd psiresp
```

Create a new conda environment with dependencies:

```
conda env create -f devtools/conda-envs/resp_env.yaml
conda activate psiresp
```

And build the package.

```
python setup.py install
```

### Example

For example, running a standard 2-stage restrained electrostatic potential fit (Bayly et al., 1993) as standard in AMBER 
(implemented as RESP-A1 in R.E.D.):

```python
   import psi4
   import psiresp

   mol = psi4.Molecule.from_string("""\
      10
      ! from R.E.D. examples/2-Dimethylsulfoxide
      C  3.87500   0.67800  -8.41700
      H  3.80000   1.69000  -8.07600
      H  3.40600   0.02600  -7.71100
      H  3.38900   0.58300  -9.36600
      S  5.35900   0.29300  -8.55900
      O  5.46000  -1.05900  -9.01400
      C  6.05500   0.43000  -7.19300
      H  7.08700   0.16300  -7.29000
      H  5.98000   1.44100  -6.85300
      H  5.58200  -0.21900  -6.48500""", fix_com=True,
      fix_orientation=True)

   r = psiresp.Resp(mol,
                    charge=0,  # overall charge of the molecule
                    multiplicity=1,  # overall multiplicity
                    name="dmso",  # name -- affects directory paths
                    stage_2=True,  # run 2-stage RESP
                    restraint=True,  # restrain ESP fit
                    hyp_a1=0.0005,  # hyperbola restraints for stage 1
                    hyp_a2=0.001,  # hyperbola restraints for stage 2
                    qm_method="hf",
                    qm_basis_set="6-31g*",
                    charge_constraint_options=dict(
                       charge_equivalences=[
                          # constrain first (atom 1) and second (atom 7) carbons to same charge
                          (1, 7),
                          # constrain all Hs to the same charge
                          (2, 3, 4, 8, 9, 10),
                          ],
                       charge_constraints=[
                          # constrain the S (atom 5) and O (atom 6) charges to sum to -0.19617
                          (-0.19617, [5, 6]),
                          ],
                       ),
                    # generate 2 orientations per conformer
                    conformer_options=dict(n_reorientations=2),
                    )
   charges = r.run()

```
Alternatively, use the preconfigured RespA1 class in ``psiresp.configs``.

```python

   r = psiresp.RespA1(mol, charge=0, multiplicity=1, name="dmso")
   r.charge_constraint_options.add_charge_equivalence(atom_ids=[1, 7])
   r.charge_constraint_options.add_charge_equivalence(atom_ids=[2, 3, 4, 8, 9, 10])
   r.charge_constraint_options.add_charge_constraint(charge=-0.19617, atom_ids=[5, 6])
   r.conformer_options.n_reorientations = 2

   charges = r.run()
```

### Pre-configured models

Each of these comes with a MultiResp counterpart.

**Probably work:**

* RespA1
* RespA2
* EspA1

**Hard to test, maybe work:**

* EspA2 (Psi4 minimises to quite a different geometry with HF/STO-3G compared to GAMESS)

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
