from typing import List, Dict, Optional

import numpy as np
import qcelemental as qcel
import qcengine as qcng

from . import base, qcutils, rdutils, orutils
from .conformer import Conformer
from .moleculebase import BaseMolecule


class ConformerGenerationOptions(base.Model):

    n_conformer_pool: int = 4000
    n_max_conformers: int = 8
    rms_tolerance: float = 0.05
    energy_window: float = 15
    clear_existing_orientations: bool = True


class Molecule(BaseMolecule):
    charge: float = 0
    conformers: List = []
    conformer_generation_options: ...

    stage_1_unrestrained_charges: Optional[np.ndarray] = None
    stage_1_restrained_charges: Optional[np.ndarray] = None
    stage_2_unrestrained_charges: Optional[np.ndarray] = None
    stage_2_restrained_charges: Optional[np.ndarray] = None

    keep_original_orientation: bool = Field(
        default=False,
        description="Whether to use the original orientation of the conformer."
    )

    reorientations: List[Tuple[int, int, int]] = Field(
        default_factory=list,
        description=("Specific rigid-body reorientations to generate. Each"
                     "number in the tuple represents an atom. The first atom"
                     "in the tuple becomes the new origin; the second defines"
                     "the x-axis; and the third defines the xy plane. This is"
                     "indexed from 0.")
    )
    translations: List[Tuple[float, float, float]] = Field(
        default_factory=list,
        description=("Specific translations to generate. Each item is a tuple of "
                     "(x, y, z) coordinates. The entire molecule is translated "
                     "by these coordinates. ")
    )
    rotations: List[Tuple[int, int, int]] = Field(default_factory=list,
                                                  description=("Specific rigid-body rotations to generate. Each"
                                                               "number in the tuple represents an atom. The first and"
                                                               "second atoms in the tuple define a vector parallel to the"
                                                               "x-axis, and the third atom defines a plane parallel to the"
                                                               "xy plane. This is indexed from 0.")
                                                  )

    def generate_transformations(self,
                                 n_rotations: int = 0,
                                 n_reorientations: int = 0,
                                 n_translations: int = 0,
                                 rotations: list = []):
        n_max = max(n_rotations, n_reorientations)
        if n_max:
            combinations = self.generate_atom_combinations(n_max)
            self.rotations.extend(combinations[:n_rotations])
            self.reorientations.extend(combinations[:n_reorientations])
        self.translations.extend((np.random.rand(n_translations, 3) - 0.5) * 10)

    @property
    def transformations(self):
        return [self.reorientations, self.rotations, self.translations]

    @property
    def n_orientations(self):
        return sum(map(len(self.transformations))) + int(self.keep_original_orientation)

    def generate_orientation_coordinates(self, coordinates=None):
        if coordinates is None:
            coordinates = self.coordinates

        orientations = []
        if self.keep_original_orientation:
            orientations.append(coordinates)

        for indices in self.reorientations:
            orientations.append(orutils.rigid_reorient(*indices, coordinates))

        for indices in self.rotations:
            orientations.append(orutils.rigid_rotate(*indices, coordinates))

        for shift in self.translations:
            orientations.append(coordinates + shift)

        return np.array(orientations)

    def generate_conformers(self):
        kwargs = self.conformer_generation_options.dict()
        kwargs.pop("clear_existing_orientations", None)
        coords = rdutils.generate_diverse_conformer_coordinates(self.qcmol, **kwargs)
        for coord in coords:
            self.add_conformer_with_coordinates(coord)

    def add_conformer_with_coordinates(self, coordinates):
        qcmol = self.qcmol_with_coordinates(coordinates)
        self.conformers.append(Conformer(qcmol=qcmol))

    def generate_orientations(self, grid_options):
        for conf in self.conformers:
            if self.conformer_generation_options.clear_existing_orientations:
                conf.orientations = []
            coords = self.generate_orientation_coordinates(conf.qcmol.geometry)
            for coord in coords:
                conf.add_orientation_with_coordinates(coord, grid_options=grid_options)


class Atom(base.Model):
    atom_index: int
    molecule: Molecule

    def __hash__(self):
        return hash((self.molecule, self.atom_index))
