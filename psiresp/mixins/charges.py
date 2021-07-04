from typing import Optional, List

import numpy as np
from pydantic import PrivateAttr, Field


from .. import base
from .charge_constraints import ChargeConstraintOptions
from .resp_base import RespStage


class RespCharges(base.Model):
    """Self-contained class wrapping RespStageOptions and ChargeConstraintOptions
    to solve RESP charges with charge constraints

    Parameters
    ----------
    resp_stage_options: RespStageOptions
        This contains the parameters for a particular stage of RESP fitting
    charge_options: ChargeConstraintOptions
        This contains the charge constraints pertinent to this stage of
        RESP fitting
    symbols: list of str
        Element symbols of the atoms to be fitted. Required to determine
        which atoms are Hs

    Attributes
    ----------
    resp_stage_options: RespStageOptions
        This contains the parameters for a particular stage of RESP fitting
    charge_options: ChargeConstraintOptions
        This contains the charge constraints pertinent to this stage of
        RESP fitting
    symbols: list of str
        Element symbols of the atoms to be fitted. Required to determine
        which atoms are Hs
    charges: numpy.ndarray of floats or None
        Overall target charges, if computed
    restrained_charges: numpy.ndarray of floats or None
        Restrained charges, if computed
    unrestrained_charges: numpy.ndarray of floats or None
        Unrestrained charges, if computed
    n_atoms: int
        Number of atoms
    """
    resp_stage_options: RespStage = RespStage()
    charge_options: ChargeConstraintOptions = ChargeConstraintOptions()
    symbols: List[str] = []
    _unrestrained_charges: Optional[np.ndarray] = PrivateAttr(default=None)
    _restrained_charges: Optional[np.ndarray] = PrivateAttr(default=None)

    @property
    def n_atoms(self):
        return len(self.symbols)

    @property
    def restrained_charges(self):
        if self._restrained_charges is not None:
            return self._restrained_charges[:self.n_atoms]

    @property
    def unrestrained_charges(self):
        if self._unrestrained_charges is not None:
            return self._unrestrained_charges[:self.n_atoms]

    @property
    def charges(self):
        restrained = self.restrained_charges
        if restrained is None:
            return self.unrestrained_charges
        return restrained

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

        a, b = self.charge_options.get_constraint_matrix(a_matrix, b_matrix)
        print(a)
        np.savetxt("test.txt", a, fmt="%6.3f")
        # print(a)
        # print(a.toarray())
        q1 = self.resp_stage_options._solve_a_b(a, b)
        self._unrestrained_charges = q1
        if self.resp_stage_options.restrained:
            q2 = self.resp_stage_options.iter_solve(q1, self.symbols, a, b)
            self._restrained_charges = q2
        return self.charges
