import warnings

import numpy as np

from .. import base
from .charge import ChargeConstraintOptions
from .conformer import ConformerOptions, ConformerGenerator


class BaseRespOptions(base.Model):
    restrained: bool = True
    hyp_b: float = 0.1
    ihfree: bool = True
    resp_convergence_tol: float = 1e-6
    resp_max_iter: int = 500
    stage_2: bool = False


class RespStageOptions(BaseRespOptions):
    hyp_a: float = 0.0005

    def get_mask_indices(self, symbols):
        symbols = np.asarray(symbols)
        mask = np.ones_like(symbols, dtype=bool)
        if self.ihfree:
            mask[np.where(symbols == "H")[0]] = False
        return np.where(mask)[0]

    @staticmethod
    def _solve_a_b(a, b):
        from scipy.sparse.linalg import spsolve
        return spsolve(a, b)

    def iter_solve(self, charges, symbols, a_matrix, b_matrix):
        if not self.hyp_a:  # i.e. no restraint
            return charges

        mask = self.get_mask_indices(symbols)
        n_atoms = len(symbols)
        b2 = self.hyp_b ** 2
        n_iter, delta = 0, 2 * self.resp_convergence_tol
        while (delta > self.resp_convergence_tol
               and n_iter < self.resp_max_iter):
            q_last = charges.copy()
            a_iter = a_matrix.copy()
            a_iter[mask] += self.hyp_a / np.sqrt(charges[mask] ** 2 + b2)
            charges = self._solve_a_b(a_iter, b_matrix)
            delta = np.max(np.abs(charges - q_last)[:n_atoms])
            n_iter += 1

        if delta > self.resp_convergence_tol:
            warnings.warn("Charge fitting did not converge to "
                          f"resp_convergence_tol={self.resp_convergence_tol} "
                          f"with resp_max_iter={self.resp_max_iter}")
        return charges


class RespOptions(BaseRespOptions, mixins.GridMixin, mixins.QMMixin):
    """
    Resp options

    Parameters
    ----------
    charge: int (optional)
        overall charge of the molecule.
    multiplicity: int (optional)
        multiplicity of the molecule
    charge_constraint_options: psiresp.options.ChargeConstraintOptions (optional)
        charge constraints and charge equivalence constraints
    conformer_options: psiresp.options.ConformerOptions (optional)
        Default arguments for creating a conformer for this object
    conformer_generator: psiresp.options.ConformerGenerator (optional)
        class to generate conformer geometries
    """
    hyp_a1: float = 0.0005
    hyp_a2: float = 0.001
    stage_2: bool = False
    charge: int = 0
    multiplicity: int = 1

    charge_constraint_options: ChargeConstraintOptions = ChargeConstraintOptions()
