# PsiRESP
😪-RESP

[//]: # "Badges"

[![Travis Build Status](https://travis-ci.com/lilyminium/psiresp.svg?branch=master)](https://travis-ci.com/lilyminium/psiresp)
[![codecov](https://codecov.io/gh/lilyminium/psiresp/branch/master/graph/badge.svg)](https://codecov.io/gh/lilyminium/psiresp/branch/master)
[![Documentation Status](https://readthedocs.org/projects/psiresp/badge/?version=latest)](https://psiresp.readthedocs.io/en/latest/?badge=latest)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/lilyminium/psiresp.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/lilyminium/psiresp/context:python)
      

A RESP plugin for Psi4.

**Things that probably work:**

* standard 2-stage RESP, gas phase HF/6-31G* with MSK radii (convenience class: psiresp.Multi/RespA1)
* intermolecular charge constraints (use the MultiResp classes for this)
* intramolecular charge constraints (use the Resp classes for this)

**Things that maybe work?**

* 1-stage ESP, B3LYP/6-31G* in implicit solvent with MSK radii. Unknown point density? (convenience class: psiresp.Multi/ATBResp)
* RESP2: PW6B95/aug-cc-pV(D+d)Z in vacuum and solvent with Bondi radii, density=2.5 (convenience class: psiresp.Multi/Resp2)

### Example

For example, running a standard 2-stage restrained electrostatic potential fit (Bayly et al., 1993) as standard in AMBER 
(implemented as RESP-A1 in R.E.D.):

```python
   import psi4
   import psiresp

   mol = psi4.Molecule.from_string("""10
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

   r = psiresp.Resp.from_molecules([mol], charge=0, name='dmso')
   charges = r.run(stage_2=True,  # run stage 2
                   opt=True,  # geometry optimize first
                   hyp_a1=0.0005, # hyperbola restraints
                   hyp_a2=0.001,
                   restraint=True,  # restrain
                   method='hf',
                   basis='6-31g*',
                   equal_methyls=True,  # restrain methyl carbons to have the same charge
                   n_orient=2)  # automatically generate 2 molecules
```
Alternatively, use the preconfigured RespA1 class in ``psiresp.configs``.

```python
   import psi4
   import psiresp

   mol = psi4.Molecule.from_string("""10
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

   r = psiresp.RespA1.from_molecules([mol], charge=0, name='dmso')
   charges = r.run(opt=True,  # geometry optimize first
                   equal_methyls=True,  # restrain methyl carbons to have the same charge
                   n_orient=2)  # automatically generate 2 molecules
```

### Pre-configured models

Each of these comes with a MultiResp counterpart, although MultiATBResp and MultiResp2 are entirely untested due to the 
original sources not supporting intermolecular charge constraints.

**Probably work:**

* RespA1
* RespA2
* EspA1

**Hard to test, maybe work:**

* EspA2 (Psi4 minimises to quite a different geometry with HF/STO-3G compared to GAMESS)
* Resp2 (Ethanol example works; others can't say for sure, optimisation at PW6B95/aug-cc-pV(D+d)Z is so slow)
* ATBResp (Paper pretty light on details; charges produced by ATB now differ from original 2011 paper)

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