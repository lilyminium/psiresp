import itertools
from typing import List, TYPE_CHECKING

import numpy as np
import qcelemental as qcel

from . import base, qcutils


def generate_atom_combinations(symbols: List[str]):
    """Yield combinations of atom indices for transformations

    The method first yields combinations of 3 heavy atom indices.
    Each combination is followed by its reverse. Once the heavy atoms
    are exhausted, the heavy atoms then get combined with the hydrogens.

    Parameters
    ----------
    symbols: list of str
        List of atom elements

    Examples
    --------

    ::

        >>> symbols = ["H", "C", "C", "O", "N"]
        >>> comb = generate_atom_combinations(symbols)
        >>> next(comb)
        (1, 2, 3)
        >>> next(comb)
        (3, 2, 1)
        >>> next(comb)
        (1, 2, 4)
        >>> next(comb)
        (4, 2, 1)

    """
    symbols = np.asarray(symbols)
    is_H = symbols == "H"
    h_atoms = list(np.flatnonzero(is_H))
    heavy_atoms = list(np.flatnonzero(~is_H) + 1)
    seen = set()

    for comb in itertools.combinations(heavy_atoms, 3):
        seen.add(comb)
        yield comb
        yield comb[::-1]

    for comb in itertools.combinations(heavy_atoms + h_atoms, 3):
        if comb in seen:
            continue
        seen.add(comb)
        yield comb
        yield comb[::-1]


class BaseMolecule(base.Model):
    qcmol: qcel.models.Molecule

    def qcmol_with_coordinates(self, coordinates):
        return qcutils.qcmol_with_coordinates(self.qcmol, coordinates)

    @property
    def n_atoms(self):
        return self.coordinates.shape[0]

    @property
    def coordinates(self):
        return self.qcmol.geometry

    def __hash__(self):
        return hash((type(self), self.qcmol.get_hash()))

    def _get_qcmol_repr(self):
        qcmol_attrs = [f"{x}={getattr(self.qcmol, x)}" for x in ["name"]]
        return ", ".join(qcmol_attrs)

    def generate_atom_combinations(self, n_combinations=None):
        atoms = generate_atom_combinations(self.qcmol.symbols)
        if n_combinations is None or n_combinations < 0:
            return atoms

        return [next(atoms) for i in range(n_combinations)]
