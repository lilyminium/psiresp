import itertools
from typing import List, Tuple, Dict

import psi4
import numpy as np
import rdkit

from . import base, utils


class OrientationGenerator(base.Model):
    """Options for generating orientations for a conformer

    Parameters
    ----------
    n_reorientations: int
        Number of rigid-body reorientations to generate
    reorientations: list of tuples of 3 atom IDs
        Specific rigid-body reorientations to generate. Each
        number in the tuple represents an atom. The first atom
        in the tuple becomes the new origin; the second defines
        the x-axis; and the third defines the xy plane. This is
        indexed from 1, such that (3, 1, 4) refers to the third,
        first and fourth atoms respectively.
    n_rotations: int
        Number of rotations to generate
    rotations: list of tuples of 3 atom IDs
        Specific rigid-body rotations to generate. Each
        number in the tuple represents an atom. The first and
        second atoms in the tuple define a vector parallel to the
        x-axis, and the third atom defines a plane parallel to the
        xy plane. This is indexed from 1,
        such that (3, 1, 4) refers to the third,
        first and fourth atoms respectively.
    n_translations: int
        Number of translations to generate
    translations: list of tuples of 3 floats
        Specific translations to generate. Each item is a tuple of
        (x, y, z) coordinates. The entire molecule is translated
        by these coordinates.
    keep_original: bool
        Whether to keep the original conformation in the conformer.
    """
    n_reorientations: int = 0
    reorientations: List[Tuple[int, int, int]] = []
    n_translations: int = 0
    translations: List[Tuple[float, float, float]] = []
    n_rotations: int = 0
    rotations: List[Tuple[int, int, int]] = []
    keep_original: bool = True
    name_template: str = "{conformer.name}_{counter:03d}"

    @property
    def transformations(self):
        return [self.reorientations, self.rotations, self.translations]

    @property
    def n_specified_transformations(self):
        return sum(map(len, self.transformations))

    @property
    def n_transformations(self):
        return sum([self.n_rotations, self.n_translations, self.n_reorientations])

    @staticmethod
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
            >>> comb = OrientationOptions.generate_atom_combinations(symbols)
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
        h_atoms = list(np.flatnonzero(is_H) + 1)
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

    def generate_transformations(self, symbols: List[str]):
        """Generate atom combinations and coordinates for transformations.

        This is used to create the number of transformations specified
        by n_rotations, n_reorientations and n_translations if these
        transformations are not already given.

        Parameters
        ----------
        symbols: list of elements
        """
        for kw in ("reorientations", "rotations"):
            n = max(self[f"n_{kw}"] - len(self[kw]), 0)
            target = self[f"n_{kw}"]
            combinations = self.generate_atom_combinations(symbols)
            while len(self[kw]) < target:
                self[kw].append(next(combinations))

        n_trans = self.n_translations - len(self.translations)
        if n_trans > 0:
            new_translations = (np.random.rand(n_trans, 3) - 0.5) * 10
            self.translations.extend(new_translations)

    @staticmethod
    def id_to_indices(atom_ids: List[int]) -> List[int]:
        """Convert atom numbers (indexed from 1) to indices (indexed from 0)

        This also works with negative atom numbers, where -1 is the last item.

        Parameters
        ----------
        atom_ids: list of ints
        """
        return [a - 1 if a > 0 else a for a in atom_ids]

    def to_indices(self) -> Dict[str, list]:
        """Return stored transformations as indices"""
        dct = {"translations": self.translations}
        dct["reorientations"] = [self.id_to_indices(x) for x in self.reorientations]
        dct["rotations"] = [self.id_to_indices(x) for x in self.rotations]
        return dct

    def get_transformed_coordinates(self,
                                    symbols: List[str],
                                    coordinates: np.ndarray,
                                    ) -> List[np.ndarray]:
        self.generate_transformations(symbols)

        transformed = []
        for reorient in self.reorientations:
            indices = id_to_indices(*reorient)
            new = utils.orientation.orient_rigid(*indices, coordinates)
            transformed.append(new)

        for rotate in self.rotations:
            indices = id_to_indices(*rotate)
            new = utils.orientation.rotate_rigid(*indices, coordinates)
            transformed.append(new)

        for translate in self.translations:
            transformed.append(coordinates + translate)

        return transformed

    def format_name(self, **kwargs):
        return self.name_template.format(**kwargs)


class ConformerGenerator(base.Model):
    conformer_geometries: List[np.ndarray] = []
    max_generated_conformers: int = 0
    min_conformer_rmsd: float = 1.5
    minimize_conformer_geometries: bool = False
    minimize_max_iter: int = 2000
    keep_original_resp_geometry: bool = True
    name_template: str = "{resp.name}_{counter:03d}"

    def generate_conformer_geometries(self, psi4mol: psi4.core.Molecule):
        rdmol = rdutils.rdmol_from_psi4mol(psi4mol)
        self._generate_conformers_from_rdmol(rdmol)

    def _generate_conformers_from_rdmol(self, rdmol: rdkit.Chem.Mol):
        if not self.keep_original_resp_geometry:
            rdmol.RemoveAllConformers()

        for coordinates in self.conformer_geometries:
            rdutils.add_conformer_from_coordinates(rdmol, coordinates)

        rdutils.generate_conformers(rdmol,
                                    n_conformers=self.max_generated_conformers,
                                    rmsd_threshold=self.min_conformer_rmsd)
        if self.minimize_conformer_geometries:
            rdutils.minimize_conformer_geometries(rdmol,
                                                  self.minimize_max_iter)
        self.conformer_geometries = rdutils.get_conformer_coordinates(rdmol)

    def format_name(self, **kwargs):
        return self.name_template.format(**kwargs)
