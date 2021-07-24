import warnings
from typing import Optional, List

import numpy as np
from pydantic import PrivateAttr, Field, validator


from .charge_constraints import ChargeConstraintOptions
from .resp_base import RespStage


class RespCharges(RespStage, ChargeConstraintOptions):
    """Self-contained class to solve RESP charges with charge constraints

    Attributes
    ----------
    charges: numpy.ndarray of floats or None
        Overall target charges, if computed
    restrained_charges: numpy.ndarray of floats or None
        Restrained charges, if computed
    unrestrained_charges: numpy.ndarray of floats or None
        Unrestrained charges, if computed
    n_atoms: int
        Number of atoms
    """
    symbols: List[str] = Field(description=("Atom element symbols. "
                                            "Required to determine Hs"))
    n_orientations: List[int] = Field(description=("Number of orientations "
                                                   "per RESP molecule"))
    _unrestrained_charges: Optional[np.ndarray] = PrivateAttr(default=None)
    _restrained_charges: Optional[np.ndarray] = PrivateAttr(default=None)
    _start_index: int = PrivateAttr(default=0)
    _charge_object: "RespCharges" = PrivateAttr()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._charge_object = self

    @validator("n_orientations")
    def validate_n_orientations(cls, v, values):
        if isinstance(v, int):
            v = [v] * len(values["symbols"])
        assert isinstance(v, list) and all(isinstance(x, int) for x in v)
        return v

    @property
    def n_atoms(self):
        return len(self.symbols)

    @property
    def restrained_charges(self):
        i = self._start_index
        j = i + self.n_atoms
        if self._charge_object._restrained_charges is not None:
            return self._charge_object._restrained_charges[i:j]

    # @restrained_charges.setter
    # def restrained_charges(self, values):
    #     if self._restrained_charges is None:
    #         self._restrained_charges = np.asarray(values)
    #     else:
    #         self._restrained_charges[:] = values

    @property
    def unrestrained_charges(self):
        i = self._start_index
        j = i + self.n_atoms
        if self._charge_object._unrestrained_charges is not None:
            return self._charge_object._unrestrained_charges[i:j]

    # @unrestrained_charges.setter
    # def unrestrained_charges(self, values):
    #     if self._restrained_charges is None:
    #         self._restrained_charges = np.asarray(values)
    #     else:
    #         self._restrained_charges[:] = values

    @property
    def charges(self):
        restrained = self.restrained_charges
        if restrained is None:
            return self.unrestrained_charges
        return restrained

    def get_mask_indices(self, symbols):
        symbols = np.asarray(symbols)
        mask = np.ones_like(symbols, dtype=bool)
        if self.ihfree:
            mask[np.where(symbols == "H")[0]] = False
        return mask

    def iter_solve(self, charges, symbols, a_matrix, b_matrix):
        if not self.hyp_a:  # i.e. no restraint
            return charges

        mask = self.get_mask_indices(symbols)
        n_atoms = len(symbols)
        diag = np.diag_indices(n_atoms)
        ix = (diag[0][mask], diag[1][mask])
        indices = np.where(mask)[0]
        n_structures = np.array(self.n_orientations)[mask]

        b2 = self.hyp_b ** 2
        n_iter, delta = 0, 2 * self.resp_convergence_tol
        while (delta > self.resp_convergence_tol
               and n_iter < self.resp_max_iter):
            q_last = charges.copy()
            a_iter = a_matrix.copy()
            increment = self.hyp_a / np.sqrt(charges[indices] ** 2 + b2) * n_structures
            a_iter[ix] += increment  # .reshape((-1, 1))
            charges = self._solve_a_b(a_iter, b_matrix)
            delta = np.max(np.abs(charges - q_last)[:n_atoms])
            n_iter += 1

        if delta > self.resp_convergence_tol:
            warnings.warn("Charge fitting did not converge to "
                          f"resp_convergence_tol={self.resp_convergence_tol} "
                          f"with resp_max_iter={self.resp_max_iter}")
        return charges

    @staticmethod
    def _solve_a_b(a, b):
        from scipy.sparse.linalg import spsolve, lsmr
        try:
            return spsolve(a, b)
        except RuntimeError as e:  # TODO: this could be slow?
            return lsmr(a, b)[0]

    def fit(self,
            a_matrix: np.ndarray,
            b_matrix: np.ndarray,
            ) -> np.ndarray:
        """Solve RESP charges with charge_options constraints

        Parameters
        ----------
        a_matrix: numpy.ndarray
        b_matrix: numpy.ndarray

        Returns
        -------
        numpy.ndarray
        """
        

        a, b = self.get_constraint_matrix(a_matrix, b_matrix)
        q1 = self._solve_a_b(a, b)
        self._unrestrained_charges = q1
        if self.restrained:
            q2 = self.iter_solve(q1, self.symbols, a, b)
            self._restrained_charges = q2
        return self.charges
