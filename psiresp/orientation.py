from typing import Optional

import numpy as np
from pydantic import Field, PrivateAttr  # , validator, root_validator
import qcelemental as qcel

from psiresp import psi4utils
from psiresp.constraint import ConstraintMatrix
from psiresp.moleculebase import BaseMolecule
from psiresp.grid import GridOptions


class OrientationEsp(BaseMolecule):
    grid: Optional[np.ndarray] = None
    esp: Optional[np.ndarray] = None
    energy: Optional[float] = None
    weight: Optional[float] = 1

    _constraint_matrix: Optional[ConstraintMatrix] = None

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
        BOHR_TO_ANGSTROM = qcel.constants.conversion_factor("bohr", "angstrom")

        displacement = self.coordinates - self.grid.reshape((-1, 1, 3))
        r_inv = BOHR_TO_ANGSTROM / np.sqrt(
            np.einsum("ijk, ijk->ij", displacement, displacement)
        )
        # r_inv = 1 / np.sqrt(
        #     np.einsum("ijk, ijk->ij", displacement, displacement)
        # )
        a = np.einsum("ij, ik->jk", r_inv, r_inv)
        b = np.einsum("i, ij->j", self.esp, r_inv)

        matrix = ConstraintMatrix.from_a_and_b(a, b)
        self._constraint_matrix = matrix
        return matrix


class Orientation(BaseMolecule):

    weight: Optional[float] = 1

    _orientation_esp: Optional[OrientationEsp] = PrivateAttr(
        default=None,
    )

    def compute_esp(self, qcrecord,
                    grid_options: GridOptions = GridOptions()):
        grid = grid_options.generate_vdw_grid(self.qcmol)
        esp = OrientationEsp(
            qcmol=self.qcmol,
            grid=grid,
            esp=psi4utils.compute_esp(qcrecord, grid),
            energy=qcrecord.properties.return_energy,
            weight=self.weight
        )
        self._orientation_esp = esp
        assert self._orientation_esp is not None
        return esp

    # grid: np.ndarray
    # esp: np.ndarray
    # qcrecord: ...
    # grid_options: ...
    # constraint_matrix: ConstraintMatrix

    # @root_validator(pre=True)
    # def check_grid_and_esp(cls, values):
    #     if values.get("constraint_matrix"):
    #         return values

    #     qcrecord = values["qcrecord"]
    #     mol = qcrecord.get_molecule()

    #     grid = values.get("grid")
    #     if grid is None:
    #         grid_options = values["grid_options"]
    #         grid = grid_options.generate_vdw_grid(mol.symbols, mol.geometry)
    #         values["grid"] = grid

    #     esp = values.get("esp")
    #     if esp is None:
    #         esp = psi4utils.compute_esp(qcrecord, grid)
    #         values["esp"] = esp

    #     matrix = ConstraintMatrix.from_grid_and_esp(mol.geometry, grid, esp)
    #     values["constraint_matrix"] = matrix
    #     return values
