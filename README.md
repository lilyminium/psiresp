# PsiRESP
😪-RESP

[//]: # "Badges"

[![Travis Build Status](https://travis-ci.com/lilyminium/psiresp.svg?branch=master)](https://travis-ci.com/lilyminium/psiresp)
[![codecov](https://codecov.io/gh/lilyminium/psiresp/branch/master/graph/badge.svg)](https://codecov.io/gh/lilyminium/psiresp/branch/master)
[![Documentation Status](https://readthedocs.org/projects/psiresp/badge/?version=latest)](https://psiresp.readthedocs.io/en/latest/?badge=latest)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/lilyminium/psiresp.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/lilyminium/psiresp/context:python)
[![Documentation Status](https://readthedocs.org/projects/psiresp/badge/?version=master)](https://psiresp.readthedocs.io/en/master/?badge=master)
      

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

```
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
                   basis='6=31g*',
                   equal_methyls=True,  # restrain methyl carbons to have the same charge
                   n_orient=2)  # automatically generate 2 molecules
```


### Copyright

Copyright (c) 2020, Lily Wang

#### Acknowledgements

Project based on the
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.2.
