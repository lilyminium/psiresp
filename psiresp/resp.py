from typing import Optional
import warnings

import numpy as np
from pydantic import Field

from psiresp import base, charge
from psiresp.constraint import (ESPSurfaceConstraintMatrix,
                                SparseGlobalConstraintMatrix)


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


class RespOptions(BaseRespOptions):
    resp_a1: float = Field(
        default=0.0005,
        description=("scale factor of asymptote limits of hyperbola, "
                     "in the stage 1 fit"),
    )
    resp_a2: float = Field(
        default=0.001,
        description=("scale factor of asymptote limits of hyperbola, "
                     "in the stage 2 fit"),
    )

    stage_2: bool = True

    @property
    def _base_kwargs(self):
        return {k: getattr(self, k) for k in BaseRespOptions.__fields__}


class RespCharges(BaseRespOptions):
    resp_a: float = Field(default=0.0005,
                          description="scale factor of asymptote limits of hyperbola")

    _restrained_charges: Optional[np.ndarray] = None
    _unrestrained_charges: Optional[np.ndarray] = None
    _matrix: Optional[SparseGlobalConstraintMatrix] = None

    charge_constraints: charge.MoleculeChargeConstraints
    surface_constraints: ESPSurfaceConstraintMatrix

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._matrix = SparseGlobalConstraintMatrix.from_constraints(
            surface_constraints=self.surface_constraints,
            charge_constraints=self.charge_constraints,
            exclude_hydrogens=self.exclude_hydrogens,
        )

    # def solve(self):
    #     from scipy.sparse.linalg import spsolve, lsmr
    #     # matrix = self.charge_constraints.construct_constraint_matrix(self.surface_constraints,
    #     #                                                              mask=self._matrix_mask)
    #     self._matrix._solve()
    #     self._unrestrained_charges = self._matrix._charges.flatten()
    #     if not self.restrained_fit or not self.resp_a:
    #         return

    #     indices = np.where(self._matrix_mask)[0]
    #     diag = np.diag_indices(len(self.symbols))
    #     ix = (diag[0][self._matrix_mask], diag[1][self._matrix_mask])

    #     b2 = self.resp_b ** 2
    #     n_iter, delta = 0, 2 * self.convergence_tolerance
    #     charges = self._unrestrained_charges.copy()
    #     while (delta > self.convergence_tolerance
    #            and n_iter < self.max_iter):
    #         q_last = self._unrestrained_charges.copy()
    #         a_iter = self._matrix.a.copy()
    #         increment = self._resp_a / np.sqrt(charges[indices] ** 2 + b2)
    #         a_iter[ix] += increment  # .reshape((-1, 1))
    #         charges = spsolve(a_iter, self._matrix.b)
    #         delta = np.max(np.abs(charges - q_last)[:len(self.symbols)])
    #         n_iter += 1

    #     self._restrained_charges = charges

    # @property
    # def _matrix_mask(self):
    #     symbols = self.symbols
    #     mask = np.ones_like(symbols, dtype=bool)
    #     if self.exclude_hydrogens:
    #         mask[np.where(symbols == "H")[0]] = False
    #     return mask

    # @property
    # def _resp_b_squared(self):
    #     return self.resp_b ** 2

    # @property
    # def n_structure_array(self):
    #     return np.concatenate([[mol.n_orientations] * mol.n_atoms
    #                            for mol in self.molecules])

    # @property
    # def molecules(self):
    #     return self.charge_constraints.molecules

    # @property
    # def _resp_a(self):
    #     return self.resp_a * self.n_structure_array[self._matrix_mask]

    # @property
    # def symbols(self):
    #     return np.concatenate([m.qcmol.symbols
    #                            for m in self.molecules])

    def solve(self):
        self._matrix._solve()
        self._unrestrained_charges = self._matrix._charges.flatten()
        if not self.restrained_fit or not self.resp_a:
            return

        n_iter = 0
        while (self._matrix.charge_difference > self.convergence_tolerance
               and n_iter < self.max_iter):
            self._matrix._iter_solve(self.resp_a, self.resp_b)
            n_iter += 1

        if self._matrix.charge_difference > self.convergence_tolerance:
            warnings.warn("Charge fitting did not converge to "
                          f"convergence_tolerance={self.convergence_tolerance} "
                          f"with max_iter={self.max_iter}")
        self._restrained_charges = self._matrix._charges.flatten()

    @property
    def restrained_charges(self):
        if self._restrained_charges is None:
            return None
        return self.charge_constraints._index_array(self._restrained_charges)

    @property
    def unrestrained_charges(self):
        if self._unrestrained_charges is None:
            return None
        return self.charge_constraints._index_array(self._unrestrained_charges)

    @property
    def _charges(self):
        if self._restrained_charges is not None:
            return self._restrained_charges
        return self._unrestrained_charges

    @property
    def charges(self):
        return self.charge_constraints._index_array(self._charges)
