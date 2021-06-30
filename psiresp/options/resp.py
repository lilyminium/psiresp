import warnings

import numpy as np

from .base import options, OptionsBase

@options
class RespOptions(OptionsBase):
    restrained: bool = True
    hyp_a1: float = 0.0005
    hyp_a2: float = 0.001
    hyp_b: float = 0.1
    ihfree: bool = True
    resp_convergence_tol: float = 1e-6
    resp_max_iter: int = 500
    stage_2: bool = False

@options
class RespStageOptions(OptionsBase):
    """Options for computing the RESP fit
    
    Parameters
    ----------
    restrained: bool
        Whether to perform a restrained fit.

    """
    restrained: bool = True
    hyp_a: float = 0.0005
    hyp_b: float = 0.1
    ihfree: bool = True
    resp_convergence_tol: float = 1e-6
    resp_max_iter: int = 500

    @classmethod
    def from_resp_options(cls, options, hyp_a_name="hyp_a1"):
        base = cls()
        kwargs = {k: options[k] for k in base.keys() if k in options}
        kwargs["hyp_a"] = options[hyp_a_name]
        return cls(**kwargs)

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

