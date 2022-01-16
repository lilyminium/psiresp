from typing import List, Optional, Tuple, Dict
import functools
import logging

import numpy as np
from pydantic import Field

from . import base, orutils
from .conformer import Conformer, ConformerGenerationOptions
from .moleculebase import BaseMolecule

logger = logging.getLogger(__name__)


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
        description="List of psiresp.conformer.Conformers of the molecule"
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
        description=("Whether to use the original orientation of the conformer. "
                     "If `keep_original_orientation=False` but "
                     "generate_orientations() is called without specifying "
                     "specific reorientations, rotations, or translations, "
                     "this is ignored and the original conformer geometry is used.")
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

    @classmethod
    def from_smiles(cls, smiles, **kwargs):
        from . import rdutils
        rdmol = rdutils.rdmol_from_smiles(smiles)
        return cls.from_rdkit(rdmol, **kwargs)

    @classmethod
    def from_rdkit(cls, molecule, random_seed=-1, **kwargs):
        from . import rdutils
        return rdutils.molecule_from_rdkit(molecule, cls,
                                           random_seed=random_seed,
                                           **kwargs)

    def to_rdkit(self):
        from .rdutils import molecule_to_rdkit
        return molecule_to_rdkit(self)

    def to_mdanalysis(self):
        from .mdautils import molecule_to_mdanalysis
        return molecule_to_mdanalysis(self)

    @property
    def charges(self):
        for charge_prop in [
            self.stage_2_restrained_charges,
            self.stage_2_unrestrained_charges,
            self.stage_1_restrained_charges,
            self.stage_1_unrestrained_charges,
        ]:
            if charge_prop is not None:
                return charge_prop

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

    def __hash__(self):
        return hash(self.qcmol.get_hash())

    def __eq__(self, other):
        return hash(self) == hash(other)

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
    def n_conformers(self):
        return len(self.conformers)

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
        """Number of orientations in the molecule."""
        return sum(conf.n_orientations for conf in self.conformers)

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
        if not self.conformers:
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

    def generate_orientations(self, clear_existing_orientations: bool = False):
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

    def get_atoms_from_smarts(self, smiles):
        indices = self.get_smarts_matches(smiles)
        return [Atom.from_molecule(self, indices=ix) for ix in indices]

    def get_sp3_ch_indices(self) -> Dict[int, List[int]]:
        get_connectivity = None
        if self._rdmol:
            try:
                from .rdutils import get_connectivity
            except ImportError:
                pass
        if get_connectivity is None:
            try:
                from .psi4utils import get_connectivity
            except ImportError:
                def get_connectivity(x):
                    return x.qcmol.connectivity

        symbols = self.qcmol.symbols
        bonds = get_connectivity(self)
        if bonds is None:
            bonds = np.empty(0, 3)
        bonds = np.asarray(bonds)
        single_bonds = np.isclose(bonds[:, 2], np.ones_like(bonds[:, 2]))
        groups = {}
        for i in np.where(symbols == "C")[0]:
            contains_index = np.any(bonds[:, :2] == i, axis=1)
            c_bonds = bonds[contains_index & single_bonds][:, :2].astype(int)
            c_partners = c_bonds[c_bonds != i]
            if len(c_partners) == 4:
                groups[i] = c_partners[symbols[c_partners] == "H"]
        return groups


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

    @property
    def symbol(self):
        return self.molecule.qcmol.symbols[self.index]

    @property
    def position(self):
        return self.molecule.qcmol.geometry[self.index]

    @property
    def atomic_number(self):
        return self.molecule.qcmol.atomic_numbers[self.index]

    def __hash__(self):
        return hash((self.molecule, self.index))

    def __eq__(self, other):
        return self.molecule == other.molecule and self.index == other.index

    def __lt__(self, other):
        return self.index < other.index
