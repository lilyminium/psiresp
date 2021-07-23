
from io import StringIO
import os
import re
import psi4
import pathlib
import numpy as np
from numpy.testing import assert_almost_equal

from psiresp.utils import BOHR_TO_ANGSTROM
from psiresp import Orientation, Conformer, Resp


def data_dir(subpath):
    data = pathlib.Path(__file__).resolve().parent
    return data / subpath


def coordinates_from_xyzfile(file):
    return np.loadtxt(file, skiprows=2, usecols=(1, 2, 3), comments='!')


def psi4mol_from_xyzfile(file):
    with open(file, "r") as f:
        xyz = f.read()
    mol = psi4.core.Molecule.from_string(xyz, dtype="xyz", fix_com=True,
                                         fix_orientation=True)
    mol.update_geometry()
    mol.activate_all_fragments()
    return mol


def assert_coordinates_almost_equal(a, b, decimal=5):
    # psi4 can translate molecules
    diff = a[0] - b[0]
    c = a - diff
    return assert_almost_equal(c, b, decimal=decimal)


def get_angstrom_coordinates(psi4mol):
    return psi4mol.geometry().np.astype('float') * BOHR_TO_ANGSTROM


def conformer_from_psi4mol(psi4mol, **kwargs):
    resp = Resp(psi4mol=psi4mol)
    resp.add_conformer(psi4mol, **kwargs)
    return resp.conformers[-1]


def orientation_from_psi4mol(psi4mol):
    conformer = conformer_from_psi4mol(psi4mol)
    orientation = conformer.add_orientation(psi4mol)
    return orientation


def esp_from_gamess_file(file):
    bohr = np.loadtxt(file, comments='!')
    bohr[:, 1:] *= BOHR_TO_ANGSTROM
    return bohr


def charges_from_red_file(file):
    with open(file, 'r') as f:
        content = f.read()

    mols = [x.split('\n')[1:] for x in content.split('MOLECULE') if x]
    charges = [np.array([float(x.split()[4]) for x in y if x]) for y in mols]
    if len(charges) == 1:
        charges = charges[0]
    return charges
