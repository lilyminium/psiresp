from typing import List, Optional
import concurrent.futures

import numpy as np
from pydantic import PrivateAttr, Field

from .charges import RespCharges
from .resp_base import BaseRespOptions, RespStage
from .qm import QMMixin
from .grid import GridMixin
from .charge_constraints import ChargeConstraintOptions
from .conformer import ConformerOptions
from .. import utils
from ..utils.execution import run_with_executor


class RespOptions(BaseRespOptions):
    """Resp options

    Parameters
    ----------
    hyp_a1: float (optional)
        scale factor of asymptote limits of hyperbola, in the stage 1 fit
    hyp_a2: float (optional)
        scale factor of asymptote limits of hyperbola, in the stage 2 fit
    """
    hyp_a1: float = 0.0005
    hyp_a2: float = 0.001


class RespMixin(RespOptions, GridMixin, QMMixin):
    """Resp mixin for actually running the job

    Parameters
    ----------
    charge_constraint_options: psiresp.options.ChargeConstraintOptions (optional)
        charge constraints and charge equivalence constraints
    conformer_options: psiresp.options.ConformerOptions (optional)
        Default arguments for creating a conformer for this object

    """

    _stage_1_charges: Optional[RespCharges] = PrivateAttr(default=None)
    _stage_2_charges: Optional[RespCharges] = PrivateAttr(default=None)
    charge_constraint_options: ChargeConstraintOptions = Field(default_factory=ChargeConstraintOptions)
    conformer_options: ConformerOptions = Field(default_factory=ConformerOptions)

    @property
    def orientations(self):
        for conformer in self.conformers:
            for orientation in conformer.orientations:
                yield orientation

    @property
    def stage_1_charges(self):
        return self._stage_1_charges

    @property
    def stage_2_charges(self):
        return self._stage_2_charges

    @property
    def n_conformers(self):
        return len(self.conformers)

    @property
    def n_orientations(self):
        return sum(conf.n_orientations for conf in self.conformers)

    @property
    def charges(self):
        if self._stage_2_charges is not None:
            return self._stage_2_charges.charges
        try:
            return self._stage_1_charges.charges
        except AttributeError:
            return self._stage_1_charges

    def get_conformer_a_matrix(self) -> np.ndarray:
        """Average the inverse squared distance matrices
        from each conformer to generate the A matrix
        for solving Ax = B

        Returns
        -------
        numpy.ndarray
            The shape of this array is (n_atoms, n_atoms)
        """
        a_matrices = [conf.weighted_a_matrix for conf in self.conformers]
        return np.sum(a_matrices, axis=0)
        # return np.mean(a_matrices, axis=0)

    def get_a_matrix(self) -> np.ndarray:
        """Average the inverse squared distance matrices
        from each conformer to generate the A matrix
        for solving Ax = B

        Returns
        -------
        numpy.ndarray
            The shape of this array is (n_atoms + 1, n_atoms + 1)
        """
        A = np.zeros((self.n_atoms + 1, self.n_atoms + 1))
        A[:-1, :-1] = self.get_conformer_a_matrix()
        A[-1, :-1] = A[:-1, -1] = 1
        return A

    def get_conformer_b_matrix(self) -> np.ndarray:
        """Average the ESP by distance from each conformer
        to generate the B vector for solving Ax = B

        Returns
        -------
        numpy.ndarray
            The shape of this vector is (n_atoms,)
        """
        b_matrices = [conf.weighted_b_matrix for conf in self.conformers]
        return np.sum(b_matrices, axis=0)
        # return np.mean(b_matrices, axis=0)

    def get_b_matrix(self) -> np.ndarray:
        """Average the ESP by distance from each conformer
        to generate the B vector for solving Ax = B

        Returns
        -------
        numpy.ndarray
            The shape of this vector is (n_atoms + 1,)
        """
        B = np.zeros(self.n_atoms + 1)
        B[:-1] = self.get_conformer_b_matrix()
        B[-1] = self.charge
        return B

    def finalize_geometries(self,
                            executor: Optional[concurrent.futures.Executor] = None,
                            timeout: Optional[float] = None,
                            command_log: str = "finalize_geometry_commands.log"):
        """Finalize geometries for all conformers.

        This is provided as a convenience function for computing all
        geometry optimizations at once with multiple processes, using a
        concurrent.futures.ProcessPoolExecutor. If an executor is not
        provided, computations are run in serial. If ``execute_qm`` is
        False, all Psi4 input job files will be written out and
        commands for running Psi4 will be saved to ``command_log``.

        Parameters
        ----------
        executor: concurrent.futures.Executor (optional)
            Executor for running jobs
        timeout: float or int (optional)
            Timeout to wait before stopping the executor
        command_log: str (optional)
            Filename to write commands to

        Raises
        ------
        SystemExit
            If ``execute_qm`` is False
        """
        functions = [conf.finalize_geometry for conf in self.conformers]
        run_with_executor(functions, executor=executor, timeout=timeout,
                          command_log=command_log)

    def compute_esps(self,
                     executor: Optional[concurrent.futures.Executor] = None,
                     timeout: Optional[float] = None,
                     command_log: str = "compute_esp_commands.log"):
        """Finalize geometries for all conformers.

        This is provided as a convenience function for computing all
        electrostatic potentials at once with multiple processes, using a
        concurrent.futures.ProcessPoolExecutor. If an executor is not
        provided, computations are run in serial. If ``execute_qm`` is
        False, all Psi4 input job files will be written out and
        commands for running Psi4 will be saved to ``command_log``.

        Parameters
        ----------
        executor: concurrent.futures.Executor (optional)
            Executor for running jobs
        timeout: float or int (optional)
            Timeout to wait before stopping the executor
        command_log: str (optional)
            Filename to write commands to

        Raises
        ------
        SystemExit
            If ``execute_qm`` is False
        """
        for conformer in self.conformers:
            if not conformer._finalized:
                raise ValueError("Finalize conformer geometry before computing "
                                 "orientation ESPs")
        functions = [orient.compute_esp for orient in self.orientations]
        run_with_executor(functions, executor=executor, timeout=timeout,
                          command_log=command_log)

    def get_clean_charge_options(self):
        return self.charge_constraint_options.copy(deep=True)

    def generate_orientations(self):
        """Generate Orientations for all conformers"""
        for conformer in self.conformers:
            conformer.generate_orientations()

    def run(self,
            executor: Optional[concurrent.futures.Executor] = None,
            timeout: Optional[float] = None,
            geometry_command_log: str = "finalize_geometry_commands.log",
            esp_command_log: str = "finalize_geometry_commands.log",
            ) -> np.ndarray:
        """Run RESP job.

        This is the recommended way to use this class, after
        setting up all configuration options.

        The geometry optimizations and ESP computations can be
        expensive. A way to parallelize operations is provided if
        ``executor`` is a concurrent.futures.ProcessPoolExecutor.
        If an executor is not provided, computations are run in serial.
        If ``execute_qm`` is False, this function will have to be
        executed multiple times:

            * (if ``optimize_geometry=True``) to run the Psi4 jobs as laid out in ``geometry_command_log``
            * to run the Psi4 jobs as laid out in ``esp_command_log``
            * to fit the RESP charges

        The computed charges will be stored at:
            * :attr:`psiresp.Resp.stage_1_charges`
            * :attr:`psiresp.Resp.stage_2_charges` (if applicable)

        These will be :class:`psiresp.charges.RespCharges` objects, which
        will contain both restrained and unrestrained charges.

        The overall desired charges will be returned by :attr:`psiresp.Resp.charges`.

        Parameters
        ----------
        executor: concurrent.futures.Executor (optional)
            Executor for running jobs
        timeout: float or int (optional)
            Timeout to wait before stopping the executor
        geometry_command_log: str (optional)
            Filename to write geometry optimization commands to
        esp_command_log: str (optional)
            Filename to write ESP computation commands to

        Returns
        -------
        numpy.ndarray of float
            The final resulting charges

        Raises
        ------
        SystemExit
            If ``execute_qm`` is False
        """

        self._stage_1_charges = None
        self._stage_2_charges = None

        self.generate_conformers()
        self.finalize_geometries(executor=executor, timeout=timeout,
                                 command_log=geometry_command_log)
        self.generate_orientations()
        self.compute_esps(executor=executor, timeout=timeout,
                          command_log=esp_command_log)

        initial_charge_options = self.get_clean_charge_options()
        stage_1 = RespStage.from_model(self, hyp_a=self.hyp_a1)

        if self.stage_2:
            final_charge_options = initial_charge_options.copy(deep=True)
            initial_charge_options.charge_equivalences = []
        else:
            final_charge_options = initial_charge_options

        final_charge_options.add_sp3_equivalences(self.get_sp3_ch_ids())

        a_matrix = self.get_a_matrix()
        b_matrix = self.get_b_matrix()

        self._stage_1_charges = RespCharges(symbols=self.symbols, n_orientations=self.n_orientation_array,
                                            **initial_charge_options.to_kwargs(),
                                            **stage_1.to_kwargs())
        q1 = self._stage_1_charges.fit(a_matrix, b_matrix)

        if self.stage_2:
            final_charge_options.add_stage_2_constraints(q1)
            stage_2 = RespStage.from_model(self, hyp_a=self.hyp_a2)
            self._stage_2_charges = RespCharges(symbols=self.symbols, n_orientations=self.n_orientation_array,
                                                **final_charge_options.to_kwargs(),
                                                **stage_2.to_kwargs())
            self._stage_2_charges.fit(a_matrix, b_matrix)

        return self.charges
