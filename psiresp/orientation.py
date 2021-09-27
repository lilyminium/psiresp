from typing import Optional

import numpy as np
from pydantic import Field, validator, root_validator

from . import base, psi4utils
from .constraint import ConstraintMatrix
from .moleculebase import BaseMolecule


class Orientation(BaseMolecule):
    grid: ...
    esp: ...
    energy: Optional[float] = None
    grid_options: ...
    constraint_matrix: ...
    weight: ...

    def compute_esp(self, qcrecord):
        self.energy = qcrecord.properties.return_energy
        if self.grid is None:
            self.grid = self.grid_options.generate_vdw_grid(self.qcmol.symbols,
                                                            self.coordinates)
        self.esp = psi4utils.compute_esp(qcrecord, self.grid)
        self.constraint_matrix = self.construct_constraint_matrix()

    def construct_constraint_matrix(self):
        BOHR_TO_ANGSTROM = qcel.constants.conversion_factor("bohr", "angstrom")

        displacement = self.coordinates - self.grid.reshape((-1, 1, 3))
        r_inv = BOHR_TO_ANGSTROM / np.sqrt(
            np.einsum("ijk, ijk->ij", displacement, displacement)
        )
        a = np.einsum("ij, ik->jk", r_inv, r_inv)
        b = np.einsum("i, ij->j", self.esp, r_inv)
        return ConstraintMatrix.from_a_and_b(a, b)

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

    def get_boltzmann_weight(self, temperature: float = 298.15):
        joules = self.energy * qcel.constants.conversion_factor("hartree", "joules")
        kb_jk = qcel.constants.Boltzmann_constant
        return joules / (kb_jk * temperature)

    def get_weighted_matrix(self, temperature: float = 298.15):
        weight = self.get_boltzmann_weight(temperature)
        return self.constraint_matrix * (weight ** 2)
