import numpy as np
import qcelemental as qcel


def load_gamess_esp(file):
    bohr = np.loadtxt(file, comments='!')
    bohr[:, 1:] *= qcel.constants.conversion_factor("bohr", "angstrom")
    return bohr
