
from io import StringIO
import os
import re
import psi4
import numpy as np
from numpy.testing import assert_almost_equal

from psiresp.utils import BOHR_TO_ANGSTROM


def coordinates_from_xyzfile(file):
    return np.loadtxt(file, skiprows=2, usecols=(1, 2, 3), comments='!')


def psi4mol_from_xyzfile(file):
    with open(file, "r") as f:
        xyz = f.read()
    return psi4.core.Molecule.from_string(xyz, dtype="xyz")


def assert_coordinates_almost_equal(a, b, decimal=5):
    # psi4 can translate molecules
    diff = a[0] - b[0]
    c = a - diff
    return assert_almost_equal(c, b, decimal=decimal)


def get_angstrom_coordinates(psi4mol):
    return psi4mol.geometry().np.astype('float') * BOHR_TO_ANGSTROM
