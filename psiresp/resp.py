from typing import List


import numpy as np


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

    _restrained_charges = None
    _unrestrained_charges = None

    molecule_constraints: ...

    def solve(self):
        matrix = self.molecule_constraints.construct_constraint_matrix(mask=self._matrix_mask)
        matrix._solve()

        self._unrestrained_charges = matrix._charges
        if not self.restrained_fit or not self.resp_a:
            return

        n_iter += 1
        while (matrix.charge_difference > self.convergence_tolerance
               and n_iter < self.max_iter):
            matrix._iter_solve()
            n_iter += 1

        if matrix.charge_difference > self.convergence_tolerance:
            warnings.warn("Charge fitting did not converge to "
                          f"convergence_tolerance={self.convergence_tolerance} "
                          f"with max_iter={self.max_iter}")
        self._restrained_charges = matrix._charges

    @property
    def _resp_b_squared(self):
        return self.resp_b ** 2

    @property
    def n_structure_array(self):
        return np.array([*([mol.n_orientations] * mol.n_atoms)
                         for mol in self.molecules])

    @property
    def molecules(self):
        return self.molecule_constraints.molecules

    @property
    def _resp_a(self):
        return self.resp_a * self.n_structure_array[self._matrix_mask_bool]

    @property
    def symbols(self):
        return np.concatenate([m.qcmol.symbols
                               for m in self.molecules])

    @property
    def restrained_charges(self):
        if self._restrained_charges is None:
            return None
        return self.molecule_constraints._index_array(self._restrained_charges)

    @property
    def unrestrained_charges(self):
        if self._unrestrained_charges is None:
            return None
        return self.molecule_constraints._index_array(self._unrestrained_charges)

    @property
    def charges(self):
        if self.restrained_charges is not None:
            return self.restrained_charges
        return self.unrestrained_charges

    @property
    def _matrix_mask(self):
        symbols = self.symbols
        mask = np.ones_like(symbols, dtype=bool)
        if self.exclude_hydrogens:
            mask[np.where(symbols == "H")[0]] = False
        return mask
