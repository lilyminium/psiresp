from typing import Optional

import numpy as np
from pydantic import Field  # , PrivateAttr, validator, root_validator
import qcelemental as qcel

from .constraint import ESPSurfaceConstraintMatrix
from .moleculebase import BaseMolecule
from .grid import GridOptions
from .qcutils import QCWaveFunction
from .utils import require_package


class Orientation(BaseMolecule):
    """
    Class to manage one orientation of a conformer. This should
    not usually be created or interacted with by a user. Instead,
    users are expected to work primarily with
    :class:`psiresp.molecule.Molecule` or :class:`psiresp.job.Job`.
    """

    weight: Optional[float] = Field(
        default=1,
        description="How much to weight this orientation in the ESP surface constraints"
    )
    qc_wavefunction: Optional[QCWaveFunction] = None
    grid: Optional[np.ndarray] = None
    esp: Optional[np.ndarray] = None

    _constraint_matrix: Optional[ESPSurfaceConstraintMatrix] = None
    _qc_id: Optional[int] = None

    @property
    def energy(self):
        try:
            return self.qc_wavefunction.energy
        except AttributeError:
            return None

    def compute_grid(self, grid_options: GridOptions = GridOptions()):
        self.grid = grid_options.generate_grid(self.qcmol)

    def compute_esp(self):
        require_package("psi4")
        from . import psi4utils
        self.esp = psi4utils.compute_esp(self.qc_wavefunction, self.grid)
        return self.esp

    def compute_esp_from_record(self, record):
        self.qc_wavefunction = QCWaveFunction.from_qcrecord(record)
        self.compute_esp()

    @property
    def constraint_matrix(self):
        if self._constraint_matrix is None:
            self.construct_constraint_matrix()
        return self._constraint_matrix

    def get_weight(self, temperature: float = 298.15):
        if self.weight is None:
            return self.get_boltzmann_weight(temperature)
        return self.weight

    def get_boltzmann_weight(self, temperature: float = 298.15):
        joules = self.energy * qcel.constants.conversion_factor("hartree", "joules")
        kb_jk = qcel.constants.Boltzmann_constant
        return joules / (kb_jk * temperature)

    def get_weighted_matrix(self, temperature: float = 298.15):
        weight = self.get_weight(temperature=temperature)
        return self.constraint_matrix * (weight ** 2)

    def construct_constraint_matrix(self):
        displacement = self.coordinates - self.grid.reshape((-1, 1, 3))

        # r_inv should be in bohr units, even though
        # coordinates and displacement are in angstrom?
        BOHR_TO_ANGSTROM = qcel.constants.conversion_factor("bohr", "angstrom")
        r_inv = BOHR_TO_ANGSTROM / np.sqrt(
            np.einsum("ijk, ijk->ij", displacement, displacement)
        )

        a = np.einsum("ij, ik->jk", r_inv, r_inv)
        b = np.einsum("i, ij->j", self.esp, r_inv)

        matrix = ESPSurfaceConstraintMatrix.from_coefficient_matrix(a, b)
        self._constraint_matrix = matrix
        return matrix
