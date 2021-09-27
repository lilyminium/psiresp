
from typing import Optional, Dict, List, Any
import pathlib
import yaml

import numpy as np
import psi4
import qcelemental as qcel
from pydantic import Field

from psiresp.base import Model

class MatrixMask:

    def __init__(self, qcmol: qcel.models.Molecule):
        self.qcmol = qcmol

        

class BaseRespOptions(base.Model):
    """Base RESP options"""
    resp_b: float = Field(default=0.1,
                          description="Tightness of hyperbolic penalty at its minimum")
    restrained_fit: bool = Field(default=True,
                             description="Perform a restrained fit")
    exclude_hydrogens: bool = Field(
        default=True,
        description="if True, exclude hydrogens from restraint",
    )
    convergence_tolerance: float = Field(
        default=1e-6,
        description="threshold for convergence",
    )
    max_iter: int = Field(
        default=500,
        description="max number of iterations to solve constraint matrices",
    )

    def _generate_matrix_mask(self, qcmol: qcel.models.Molecule) -> np.ndarray:
        mask = np.ones_like(qcmol.symbols, dtype=bool)
        if self.exclude_hydrogens:
            mask[np.where(qcmol.symbols == "H")[0]] = False
        return mask

    def _generate_mask_indices(self, qcmol: qcel.models.Molecule):
        mask = self._generate_matrix_mask(qcmol)
        





class RespStage(BaseRespOptions):
    resp_a: float = Field(default=0.0005,
                          description="scale factor of asymptote limits of hyperbola")
    
    # def iter_solve(self, charges, symbols, a_matrix, b_matrix):
    #     n_iter = 0
    #     delta = np.inf
    #     b_sq = self.resp_b ** 2

    #     while (delta > self.convergence_tolerance and n_iter < self.max_iter):
    #         previous_charges = charges.copy()
    #         a_iter = a_matrix.copy()
    #         increment = self.resp_a / 