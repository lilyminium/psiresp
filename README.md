# PsiRESP

[//]: # "Badges"

[![Travis Build Status](https://travis-ci.com/lilyminium/psiresp.svg?branch=master)](https://travis-ci.com/lilyminium/psiresp)
[![AppVeyor Build status](https://ci.appveyor.com/api/projects/status/REPLACE_WITH_APPVEYOR_LINK/branch/master?svg=true)](https://ci.appveyor.com/project/lilyminium/psiresp/branch/master)
[![codecov](https://codecov.io/gh/lilyminium/psiresp/branch/master/graph/badge.svg)](https://codecov.io/gh/lilyminium/psiresp/branch/master)

A RESP plugin for Psi4. (pronounced 😪-RESP)

**Things that probably work:**

* standard 2-stage RESP, gas phase HF/6-31G* with MSK radii (convenience class: psiresp.Multi/RespA1)
* intermolecular charge constraints (use the MultiResp classes for this)
* intramolecular charge constraints (use the Resp classes for this)

**Things that maybe work?**

* 1-stage ESP, B3LYP/6-31G* in implicit solvent with MSK radii. Unknown point density? (convenience class: psiresp.Multi/ATBResp)
* RESP2: PW6B95/aug-cc-pV(D+d)Z in vacuum and solvent with Bondi radii, density=2.5 (convenience class: psiresp.Multi/Resp2)

### Copyright

Copyright (c) 2020, Lily Wang

#### Acknowledgements

Project based on the
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.2.
