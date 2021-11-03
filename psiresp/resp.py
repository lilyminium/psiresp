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
    """Self-contained class to solve RESP charges with charge constraints"""
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

    def solve(self):
        self._matrix._solve()
        self._unrestrained_charges = self._matrix._charges.flatten()
        if not self.restrained_fit or not self.resp_a:
            return

        n_iter = 0
        b2 = self.resp_b ** 2
        while (self._matrix.charge_difference > self.convergence_tolerance
               and n_iter < self.max_iter):
            print(self._matrix.charge_difference, self.convergence_tolerance)
            self._matrix._iter_solve(self.resp_a, self.resp_b, b2)
            n_iter += 1
        self._matrix._iter_solve(self.resp_a, self.resp_b, b2)

        print("NITER", n_iter)

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
