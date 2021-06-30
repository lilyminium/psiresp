import re
from typing import Dict, List

import psi4
import numpy as np
from numpy import typing as npt

from .constants import ANGSTROM_TO_BOHR

def get_mol_spec(psi4mol: psi4.core.Molecule) -> str:
    """Create Psi4 molecule specification from Psi4 molecule

    Parameters
    ----------
    psi4mol: psi4.core.Molecule
        Molecule

    Returns
    -------
    mol_spec: str
    """
    mol = psi4mol.create_psi4_string_from_molecule()
    # remove any clashing fragment charge/multiplicity
    pattern = r"--\n\s*\d \d\n"
    mol = re.sub(pattern, "", mol)
    return f"molecule {psi4mol.name()} {{\n{mol}\n}}\n\n"


def set_psi4mol_geometry(psi4mol: psi4.core.Molecule,
                         coordinates: npt.NDArray,
                         angstrom: bool = True):
    """
    Set geometry of `psi4mol` molecule with numpy array of coordinates

    Parameters
    ----------
    psi4mol: psi4.core.Molecule
        Molecule to modify in-place
    coordinates: np.ndarray of floats, of shape N x 3
        coordinates. Generally assumed to be angstrom
    angstrom: bool
        Whether the coordinates are in angstrom
    """
    charge = psi4mol.molecular_charge()
    multiplicity = psi4mol.molecular_charge()

    if angstrom:
        coordinates = coordinates * ANGSTROM_TO_BOHR

    mat = psi4.core.Matrix.from_array(coordinates)
    psi4mol.set_geometry(mat)
    psi4mol.set_molecular_charge(charge)
    psi4mol.set_multiplicity(multiplicity)
    psi4mol.fix_com(True)
    psi4mol.fix_orientation(True)
    psi4mol.update_geometry()


def get_sp3_ch_ids(psi4mol: psi4.core.Molecule) -> Dict[int, List[int]]:
    """Get dictionary of sp3 carbon atom number to bonded hydrogen numbers.

    These atom numbers are indexed from 1. Each key is the number of an
    sp3 carbon. The value is the list of bonded hydrogen numbers.

    Parameters
    ----------
    psi4mol: psi4.core.Molecule
        Molecule
    
    Returns
    -------
    c_h_dict: dict of {int: list of ints}
    """
    indices = np.arange(psi4mol.natom())
    symbols = np.array([psi4mol.symbol(i) for i in indices])

    groups = {}
    bonds = psi4.qcdb.parker._bond_profile(psi4mol)  # [[i, j, bond_order]]
    bonds = np.asarray(bonds)[:, :2]  # [[i, j]]
    for i in indices[symbols == "C"]:
        cbonds = bonds[np.any(bonds == i, axis=1)]
        partners = cbonds[cbonds != i]
        if len(partners) == 3:
            hs = partners[symbols[partners] == "H"]
            groups[i + 1] = list(hs + 1)
    return groups


def psi4mol_from_xyz_string(string: str) -> psi4.core.Molecule:
    return psi4.core.Molecule.from_string(string, dtype="xyz")


def psi4mol_to_xyz_string(psi4mol: psi4.core.Molecule) -> str:
    return psi4mol.to_string(dtype="xyz")

def psi4mol_to_mol_string(psi4mol: psi4.core.Molecule) -> str:
    return psi4mol.format_molecule_for_mol()

def psi4logfile_to_xyz_string(logfile: str) -> str:
    """Get geometry in XYZ format from Psi4 log file"""
    with open(logfile, "r") as f:
        contents = f.read()
    last_lines = contents.split("OPTKING Finished Execution")[-1].split("\n")
    atom_spec = []
    for line in last_lines:
        line = line.strip().split()
        if len(line) == 4:
            try:
                atom_line = [line[0]] + list(map(float, line[-3:]))
            except ValueError:
                continue
            else:
                atom_spec.append(atom_line)

    name = logfile.strip(".log")
    lines = [str(len(atom_spec)), name] + atom_spec
    txt = "\n".join(lines)
    return txt


def psi4mol_from_psi4logfile(logfile) -> psi4.core.Molecule:
    xyz = psi4logfile_to_xyz_string(logfile)
    return psi4mol_from_xyz_string(xyz)

