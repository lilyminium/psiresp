from typing import List, Optional
import concurrent.futures

import numpy as np
from pydantic import PrivateAttr, Field


from .. import base
from .conformer import ConformerOptions


class BaseRespOptions(base.Model):
    restrained: bool = True
    hyp_b: float = 0.1
    ihfree: bool = True
    resp_convergence_tol: float = 1e-6
    resp_max_iter: int = 500
    stage_2: bool = False


class RespMoleculeOptions(base.Model):

    conformer_options: ConformerOptions = Field(default_factory=ConformerOptions)
    conformer_name_template: str = "{resp.name}_{counter:03d}"
    max_generated_conformers: int = 0
    min_conformer_rmsd: float = 1.5
    minimize_conformer_geometries: bool = False
    minimize_max_iter: int = 2000
    keep_original_resp_geometry: bool = True


class RespStage(BaseRespOptions):
    hyp_a: float = 0.0005

    def get_mask_indices(self, symbols):
        symbols = np.asarray(symbols)
        mask = np.ones_like(symbols, dtype=bool)
        if self.ihfree:
            mask[np.where(symbols == "H")[0]] = False
        return mask

    @staticmethod
    def _solve_a_b(a, b):
        from scipy.sparse.linalg import spsolve
        return spsolve(a, b)

    # def iter_solve(self, charges, symbols, a_matrix, b_matrix):
    #     if not self.hyp_a:  # i.e. no restraint
    #         return charges

    #     n_atoms = len(symbols)
    #     mask = np.ones(n_atoms, dtype=bool)
    #     if self.ihfree:
    #         h_indices = np.where(symbols == "H")[0]
    #         mask[h_indices] = False
    #     diag = np.diag_indices(n_atoms)
    #     ix = (diag[0][mask], diag[1][mask])
    #     indices = np.where(mask)[0]
    #     b2 = self.hyp_b**2
    #     # n_structures = self.n_structures[mask]
    #     # print(f"Fitting with {self.maxiter}")

    #     niter, delta = 0, 2 * self.resp_convergence_tol
    #     while delta > self.resp_convergence_tol and niter < self.resp_max_iter:
    #         q_last = charges.copy()
    #         a_i = a_matrix.copy()
    #         a_i[ix] = a_matrix[ix] + self.hyp_a / np.sqrt(charges[indices]**2 + b2)  # * n_structures
    #         charges = np.linalg.lstsq(a_i, b_matrix)[0]
    #         delta = np.max((charges - q_last)[:n_atoms]**2)**0.5
    #         niter += 1
    #     print("delta", delta, niter)

    #     # if delta > self.resp_options.tol:
    #     #     err = "Charge fitting did not converge with maxiter={}"
    #     #     warnings.warn(err.format(self.resp_options.maxiter))

    #     return charges

    def iter_solve(self, charges, symbols, a_matrix, b_matrix):
        if not self.hyp_a:  # i.e. no restraint
            return charges

        mask = self.get_mask_indices(symbols)
        n_atoms = len(symbols)
        diag = np.diag_indices(n_atoms)
        ix = (diag[0][mask], diag[1][mask])
        indices = np.where(mask)[0]

        b2 = self.hyp_b ** 2
        n_iter, delta = 0, 2 * self.resp_convergence_tol
        while (delta > self.resp_convergence_tol
               and n_iter < self.resp_max_iter):
            q_last = charges.copy()
            a_iter = a_matrix.copy()
            increment = self.hyp_a / np.sqrt(charges[indices] ** 2 + b2)
            a_iter[ix] += increment  # .reshape((-1, 1))
            charges = self._solve_a_b(a_iter, b_matrix)
            delta = np.max(np.abs(charges - q_last)[:n_atoms])
            n_iter += 1

        print("delta")
        print(delta, n_iter)

        if delta > self.resp_convergence_tol:
            warnings.warn("Charge fitting did not converge to "
                          f"resp_convergence_tol={self.resp_convergence_tol} "
                          f"with resp_max_iter={self.resp_max_iter}")
        return charges
