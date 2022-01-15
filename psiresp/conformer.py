
from typing import Optional, List
from pydantic import Field

import numpy as np
import qcelemental as qcel

from . import base
from .orientation import Orientation
from .moleculebase import BaseMolecule
from .utils import require_package


class Conformer(BaseMolecule):
    """Class to manage one conformer of a molecule.

    It must hold at least one orientation.
    """
    orientations: List[Orientation] = []
    is_optimized: bool = False
    _qc_id: Optional[int] = None

    @property
    def n_orientations(self):
        return len(self.orientations)

    def add_orientation_with_coordinates(self, coordinates, units="angstrom"):
        qcmol = self.qcmol_with_coordinates(coordinates, units=units)
        self.orientations.append(Orientation(qcmol=qcmol))

    def set_optimized_geometry(self, coordinates, units="bohr"):
        cf = qcel.constants.conversion_factor(units, "bohr")
        dct = self.qcmol.dict()
        dct["geometry"] = coordinates * cf
        self.qcmol = type(self.qcmol)(**dct)
        self.is_optimized = True


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
        default=30,
        description=("Energy window (kcal/mol) within which to keep conformers. "
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
        else:
            require_package("rdkit")
            from .rdutils import generate_diverse_conformer_coordinates

            rdkwargs = self.dict()
            keep = rdkwargs.pop("keep_original_conformer")
            coords = generate_diverse_conformer_coordinates(qcmol, **rdkwargs)
            if keep:
                coords = np.concatenate([original, coords])
            return coords
