import pytest

import numpy as np
import qcelemental as qcel

try:
    import qcfractal.interface
except ImportError:
    qcfractal_is_installed = False
else:
    qcfractal_is_installed = True

requires_qcfractal = pytest.mark.skipif(not qcfractal_is_installed, reason="requires QCFractal")


def load_gamess_esp(file):
    bohr = np.loadtxt(file, comments='!')
    bohr[:, 1:] *= qcel.constants.conversion_factor("bohr", "angstrom")
    return bohr
