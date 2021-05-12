from pkg_resources import resource_filename

from io import StringIO
import os
import re
import psi4
import numpy as np

#: convert bohr to angstrom
BOHR_TO_ANGSTROM = 0.52917721092


def molfile(file):
    path = os.path.join('molecules', file)
    return resource_filename(__name__, path)


def chargefile(file):
    path = os.path.join('charges', file)
    return resource_filename(__name__, path)


def datafile(file):
    path = os.path.join('data', file)
    return resource_filename(__name__, path)


def mol_from_file(file):
    with open(molfile(file), 'r') as f:
        geom = f.read()
    mol = psi4.core.Molecule.from_string(geom, fix_com=True, fix_orientation=True)
    mol.update_geometry()
    mol.activate_all_fragments()
    return mol


def mol_from_mol2(file):
    with open(molfile(file), 'r') as f:
        atoms = f.read().split('@<TRIPOS>ATOM\n')[1]
        atoms = atoms.split('\n@<TRIPOS>')[0]
    arr = np.loadtxt(StringIO(atoms), usecols=(1, 2, 3, 4), dtype=object)
    arr[:, 0] = [re.sub(r'\d+', '', x) for x in arr[:, 0]]
    rows = [' '.join(row) for row in arr]
    xyz = '{}\n\n{}'.format(len(arr), '\n'.join(rows))
    mol = psi4.core.Molecule.from_string(xyz, fix_com=True, fix_orientation=True)
    mol.update_geometry()
    return mol


def charges_from_mol2(file):
    with open(molfile(file), 'r') as f:
        atoms = f.read().split('@<TRIPOS>ATOM\n')[1]
        atoms = atoms.split('\n@<TRIPOS>')[0]
    arr = np.loadtxt(StringIO(atoms), usecols=8)
    return arr


def coordinates_from_xyz(file):
    return np.loadtxt(molfile(file), skiprows=2, usecols=(1, 2, 3), comments='!')


def charges_from_red_file(file):
    with open(chargefile(file), 'r') as f:
        content = f.read()

    mols = [x.split('\n')[1:] for x in content.split('MOLECULE') if x]
    charges = [np.array([float(x.split()[4]) for x in y if x]) for y in mols]
    if len(charges) == 1:
        charges = charges[0]
    return charges


def charges_from_itp_file(file):
    with open(chargefile(file), 'r') as f:
        content = f.read()
    section = content.split('[ atoms ]')[1].split('; total charge')[0]
    lines = [x.strip() for x in section.split('\n')]
    lines = [x for x in lines if x and not x.startswith(';')]
    return np.array([float(x.split()[6]) for x in lines])


def grid_from_file(file):
    return np.loadtxt(datafile(file), usecols=(1, 2, 3), comments='!')


def esp_from_gamess_file(file):
    bohr = np.loadtxt(datafile(file), comments='!')
    bohr[:, 1:] *= BOHR_TO_ANGSTROM
    return bohr
