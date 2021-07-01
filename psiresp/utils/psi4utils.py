import re
import pathlib
from typing import Dict, List, Optional, Union

import psi4
import numpy as np

from . import BOHR_TO_ANGSTROM, ANGSTROM_TO_BOHR

CoordinateInputs = Union[psi4.core.Molecule, np.ndarray]


def as_coordinates(coordinates_or_psi4mol: CoordinateInputs) -> np.ndarray:
    """Get coordinates from either an array or Psi4 molecule

    Parameters
    ----------
    coordinates_or_psi4mol: numpy.ndarray or psi4.core.Molecule
        Coordinate array or molecule

    Returns
    -------
    numpy.ndarray
        Coordinates in angstrom
    """
    try:
        xyz = coordinates_or_psi4mol.geometry().np.astype("float")
    except AttributeError:
        pass
    else:
        coordinates_or_psi4mol = xyz * BOHR_TO_ANGSTROM
    return coordinates_or_psi4mol


def psi4mol_with_coordinates(psi4mol: psi4.core.Molecule,
                             coordinates_or_psi4mol: CoordinateInputs,
                             name: Optional[str] = None
                             ) -> psi4.core.Molecule:
    """Return a Psi4 molecule with given coordinates and name

    Parameters
    ----------
    psi4mol: psi4.core.Molecule
        Psi4 molecule to copy
    coordinates_or_psi4mol: numpy.ndarray or psi4.core.Molecule
        Coordinate array or molecule to obtain coordinates from
    name: str (optional)
        Name for the molecule

    Returns
    -------
    psi4.core.Molecule
        Resulting molecule
    """
    clone = psi4mol.clone()
    coordinates = as_coordinates(coordinates_or_psi4mol)
    set_psi4mol_coordinates(clone, coordinates)
    if name is not None:
        clone.set_name(name)
    return clone


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


def set_psi4mol_coordinates(psi4mol: psi4.core.Molecule,
                            coordinates: np.ndarray):
    """
    Set geometry of `psi4mol` molecule with numpy array of coordinates

    Parameters
    ----------
    psi4mol: psi4.core.Molecule
        Molecule to modify in-place
    coordinates: np.ndarray of floats, of shape N x 3
        coordinates in angstrom
    """
    charge = psi4mol.molecular_charge()
    multiplicity = psi4mol.multiplicity()
    coordinates = coordinates * ANGSTROM_TO_BOHR

    mat = psi4.core.Matrix.from_array(coordinates)
    psi4mol.set_geometry(mat)
    psi4mol.set_molecular_charge(charge)
    psi4mol.set_multiplicity(multiplicity)
    psi4mol.fix_com(True)
    psi4mol.fix_orientation(True)
    psi4mol.update_geometry()


def get_psi4mol_coordinates(psi4mol: psi4.core.Molecule) -> np.ndarray:
    return psi4mol.geometry().np.astype("float") * BOHR_TO_ANGSTROM


def get_sp3_ch_ids(psi4mol: psi4.core.Molecule,
                   increment: int = 0,
                   ) -> Dict[int, List[int]]:
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
        if len(partners) == 4:
            hs = partners[symbols[partners] == "H"]
            groups[i + 1 + increment] = list(hs + 1 + increment)
    return groups


def psi4mol_from_xyz_string(string: str) -> psi4.core.Molecule:
    """Create Psi4 molecule from string of XYZ format"""
    return psi4.core.Molecule.from_string(string, dtype="xyz")


def psi4mol_to_xyz_string(psi4mol: psi4.core.Molecule) -> str:
    """Create XYZ representation of Psi4 molecule"""
    return psi4mol.to_string(dtype="xyz")


def psi4mol_to_mol2_string(psi4mol: psi4.core.Molecule) -> str:
    """Create MOL2 representation of Psi4 molecule"""
    return psi4mol.format_molecule_for_mol()


def opt_logfile_to_xyz_string(logfile: str) -> str:
    """Get geometry in XYZ format from Psi4 optimization log file"""
    with open(logfile, "r") as f:
        contents = f.read()
    last_lines = contents.split("OPTKING Finished Execution")[-1].split("\n")
    atom_spec = []
    for line in last_lines:
        line = line.strip()
        fields = line.split()
        if len(fields) == 4:
            try:
                atom_line = [fields[0]] + list(map(float, fields[-3:]))
            except ValueError:
                continue
            else:
                atom_spec.append(line)

    name = pathlib.Path(logfile).stem
    lines = [str(len(atom_spec)), name] + atom_spec
    txt = "\n".join(lines)
    return txt


def psi4mol_from_psi4optfile(logfile) -> psi4.core.Molecule:
    """Create Psi4 molecule from optimized geometry of log file"""
    xyz = opt_logfile_to_xyz_string(logfile)
    return psi4mol_from_xyz_string(xyz)
