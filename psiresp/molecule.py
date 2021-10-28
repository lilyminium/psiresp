from typing import List, Optional, Tuple
import functools
import logging

import numpy as np
import qcelemental as qcel
from pydantic import Field

from . import base, rdutils, orutils
from .conformer import Conformer
from .moleculebase import BaseMolecule

logger = logging.getLogger(__name__)


class ConformerGenerationOptions(base.Model):
    """Options for generating conformers"""

    n_conformer_pool: int = Field(
        default=4000,
        description="Number of initial conformers to generate"
    )
    n_max_conformers: int = Field(
        default=0,
        description="Maximum number of conformers to keep"
    )
    rms_tolerance: float = Field(
        default=0.05,
        description="RMS tolerance for pruning conformers"
    )
    energy_window: float = Field(
        default=15,
        description=("Energy window within which to keep conformers. "
                     "This is the range from the lowest energetic conformer"),
    )
    keep_original_conformer: bool = Field(
        default=True,
        description="Whether to keep the original conformer in the molecule"
    )

    def generate_coordinates(self, qcmol: qcel.models.Molecule) -> np.ndarray:
        """Generate conformer coordinates in angstrom"""
        original = np.array([qcmol.geometry])
        original *= qcel.constants.conversion_factor("bohr", "angstrom")
        if not self.n_max_conformers:
            return original

        rdkwargs = self.dict()
        keep = rdkwargs.pop("keep_original_conformer")
        coords = rdutils.generate_diverse_conformer_coordinates(qcmol,
                                                                **rdkwargs)
        if keep:
            coords = np.concatenate([original, coords])
        return coords


class Molecule(BaseMolecule):
    """Class to manage a molecule"""
    charge: Optional[int] = Field(
        default=None,
        description=("Charge to apply to the molecule. "
                     "If `charge=None`, the charge is taken to be "
                     "the molecular_charge on the QCElemental molecule")
    )
    multiplicity: Optional[int] = Field(
        default=None,
        description=("Multiplicity to apply to the molecule. "
                     "If `multiplicity=None`, the multiplicity is taken to be "
                     "the molecular_multiplicity on the QCElemental molecule")
    )
    optimize_geometry: bool = Field(
        default=False,
        description="Whether to optimize the geometry of conformers",
    )
    conformers: List[Conformer] = Field(
        default_factory=list,
        description="List of `psiresp.conformer.Conformer`s of the molecule"
    )
    conformer_generation_options: ConformerGenerationOptions = Field(
        default_factory=ConformerGenerationOptions,
        description="Conformer generation options",
    )

    stage_1_unrestrained_charges: Optional[np.ndarray] = Field(
        default=None,
        description=("Stage 1 unrestrained charges. "
                     "These are typically assigned from a "
                     ":class:`psiresp.job.Job`.")
    )
    stage_1_restrained_charges: Optional[np.ndarray] = Field(
        default=None,
        description=("Stage 1 restrained charges. "
                     "These are typically assigned from a "
                     ":class:`psiresp.job.Job`.")
    )
    stage_2_unrestrained_charges: Optional[np.ndarray] = Field(
        default=None,
        description=("Stage 2 unrestrained charges. "
                     "These are typically assigned from a "
                     ":class:`psiresp.job.Job`.")
    )
    stage_2_restrained_charges: Optional[np.ndarray] = Field(
        default=None,
        description=("Stage 2 restrained charges. "
                     "These are typically assigned from a "
                     ":class:`psiresp.job.Job`.")
    )

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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.charge is None:
            self.charge = self.qcmol.molecular_charge
        else:
            self.qcmol.__dict__["molecular_charge"] = self.charge
        if self.multiplicity is None:
            self.multiplicity = self.qcmol.molecular_multiplicity
        else:
            self.qcmol.__dict__["molecular_multiplicity"] = self.multiplicity

    def __repr__(self):
        qcmol_repr = self._get_qcmol_repr()
        n_confs = len(self.conformers)
        return f"{self._clsname}({qcmol_repr}, charge={self.charge}) with {n_confs} conformers"

    def generate_transformations(self,
                                 n_rotations: int = 0,
                                 n_reorientations: int = 0,
                                 n_translations: int = 0):
        """Automatically generate the atom combinations for
        rotations and reorientations, as well as translation
        vectors.

        Atom combinations are generated first by iterating over
        combinations of heavy atoms, and then incorporating hydrogens.
        """
        n_max = max(n_rotations, n_reorientations)
        if n_max:
            combinations = self.generate_atom_combinations(n_max)
            self.rotations.extend(combinations[:n_rotations])
            self.reorientations.extend(combinations[:n_reorientations])
        self.translations.extend((np.random.rand(n_translations, 3) - 0.5) * 10)

    @property
    def transformations(self):
        """All transformations"""
        return [self.reorientations, self.rotations, self.translations]

    @property
    def n_transformations(self):
        "Number of transformations"
        return sum(map(len, self.transformations))

    @property
    def n_orientations(self):
        """Number of orientations that should exist.
        This may not correspond to the number of orientations actually in
        the conformer, if generate_orientations has not been called yet."""
        return len(self.conformers) * (self.n_transformations + int(self.keep_original_orientation))

    def generate_orientation_coordinates(
        self,
        coordinates: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Generate coordinates for orientations in angstrom"""
        if coordinates is None:
            coordinates = self.coordinates

        orientations = []
        if self.keep_original_orientation:
            orientations.append(coordinates)

        for indices in self.reorientations:
            orientations.append(orutils.rigid_orient(*indices, coordinates))

        for indices in self.rotations:
            orientations.append(orutils.rigid_rotate(*indices, coordinates))

        for shift in self.translations:
            orientations.append(coordinates + shift)

        return np.array(orientations)

    def generate_conformers(self):
        """Generate conformers"""
        coords = self.conformer_generation_options.generate_coordinates(self.qcmol)
        for coord in coords:
            self.add_conformer_with_coordinates(coord)
        if not self.conformers:
            self.add_conformer_with_coordinates(self.coordinates)

    def add_conformer(self, conformer=None, **kwargs):
        if conformer is not None and isinstance(conformer, Conformer):
            if not np.equals(conformer.qcmol.symbols, self.qcmol.symbols):
                raise ValueError("Conformer molecule does not match. "
                                 f"Conformer symbols: {conformer.qcmol.symbols}, "
                                 f"Molecule symbols: {self.qcmol.symbols}")
        else:
            conformer = Conformer(**kwargs)
        self.conformers.append(conformer)

    def add_conformer_with_coordinates(self, coordinates, units="angstrom"):
        """Add a new conformer with specified coordinates.

        No checking is done to ensure that the
        conformer does not already exist in the molecule.
        """
        qcmol = self.qcmol_with_coordinates(coordinates, units=units)
        self.conformers.append(Conformer(qcmol=qcmol))

    def generate_orientations(self, clear_existing_orientations: bool = True):
        """Generate Orientation objects for each conformer."""
        if not self.conformers:
            self.generate_conformers()
        for conf in self.conformers:
            if clear_existing_orientations:
                conf.orientations = []
            coords = self.generate_orientation_coordinates(conf.coordinates)
            for coord in coords:
                conf.add_orientation_with_coordinates(coord)
            if not len(conf.orientations):
                conf.add_orientation_with_coordinates(conf.coordinates)


@functools.total_ordering
class Atom(base.Model):
    """Class to manage atoms for charge constraints"""
    index: int
    molecule: Molecule

    @classmethod
    def from_molecule(cls, molecule, indices=[]):
        if not isinstance(indices, (list, tuple, np.ndarray)):
            indices = [indices]
        return [cls(molecule=molecule, index=i) for i in indices]

    def __hash__(self):
        return hash((self.molecule, self.index))

    def __eq__(self, other):
        return self.molecule == other.molecule and self.index == other.index

    def __lt__(self, other):
        return self.index < other.index
