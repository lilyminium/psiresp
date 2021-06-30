import logging
from dataclasses import dataclass, field
import concurrent.futures

import numpy as np

from .conformer import Conformer
from . import utils, base, psi4utils, exceptions, constants, mixins
from .options import ChargeOptions, RespOptions, RespStageOptions, QMOptions
from .charges import RespCharges

logger = logging.getLogger(__name__)


@dataclass
class Resp(options.RespOptions, mixins.GridMixin, mixins.QMMixin, mixins.MoleculeMixin):

    charge: int = 0
    multiplicity: int = 1

    charge_constraint_options: ChargeOptions = ChargeOptions()
    conformer_options: ConformerOptions = ConformerOptions()
    conformer_generator: ConformerGenerator = ConformerGenerator()

    
    def __post_init__(self):
        super().__post_init__()
        self.conformers = []
        self.stage_1_charges = None
        self.stage_2_charges = None

    @property
    def n_conformers(self):
        return len(self.conformers)

    def add_conformer(self,
                      coordinates_or_psi4mol: psi4utils.CoordinateInputs,
                      name: Optional[str] = None,
                      **kwargs) -> Conformer:
        """Create Conformer from Psi4 molecule or coordinates and add

        Parameters
        ----------
        coordinates_or_psi4mol: numpy.ndarray of coordinates or psi4.core.Molecule
            An array of coordinates or a Psi4 Molecule. If this is a molecule,
            the molecule is copied before creating the Conformer.
        name: str (optional)
            Name of the conformer. If not provided, one will be generated
            from the name template in the conformer_generator
        **kwargs:
            Arguments used to construct the Conformer. If not provided,
            the default specification given in ``self.conformer_options``
            will be used.

        Returns
        -------
        conformer: Conformer
        """
        if name is None:
            counter = len(self.conformers) + 1
            name = self.conformer_options.format_name(resp=self,
                                                      counter=counter)
        mol = psi4utils.psi4mol_with_coordinates(self.psi4mol,
                                                 coordinates_or_psi4mol,
                                                 name=name)
        default_kwargs = self.conformer_options.dict()
        default_kwargs.update(kwargs)
        conf = Conformer(resp=self, psi4mol=mol, name=name, **default_kwargs)
        self.conformers.append(conf)
        return conf

    def generate_conformers(self):
        """Generate conformers from settings in conformer_generator.

        If no conformers result from those settings, the geometry of the
        input Psi4 molecule to RESP is used.
        """
        self.conformers = []
        self.conformer_options.generate_conformer_geometries(self.psi4mol)
        for coordinates in self.conformer_options.conformer_geometries:
            self.add_conformer(coordinates)
        if not self.conformers:
            self.add_conformer(self.psi4mol)

    @property
    def orientations(self):
        for conformer in self.conformers:
            for orientation in conformer.orientations:
                yield orientation

    @property
    def charges(self):
        if self.stage_2_charges is not None:
            return self.stage_2_charges.charges
        try:
            return self.stage_1_charges.charges
        except AttributeError:
            return self.stage_1_charges

    def to_mda(self):
        """Create a MDAnalysis.Universe with charges
        
        Returns
        -------
        MDAnalysis.Universe
        """
        u = super().to_mda()
        if self.charges is not None:
            u.add_TopologyAttr("charges", self.charges)
        return u

    def get_conformer_a_matrix(self) -> npt.NDArray:
        """Sum the inverse squared distance matrices
        from each conformer and orientation to generate the A matrix
        for solving Ax = B

        Returns
        -------
        numpy.ndarray
            The shape of this array is (n_atoms + 1, n_atoms + 1)
        """
        A = np.zeros((self.n_atoms + 1, self.n_atoms + 1))
        for conformer in self.conformers:
            A[:-1, :-1] += conformer.weighted_a_matrix
        A /= self.n_conformers
        A[-1, :-1] = A[:-1, -1] = 1
        return A

    def get_conformer_b_matrix(self) -> npt.NDArray:
        """Sum the ESP by distance from each conformer and orientation
        to generate the B vector for solving Ax = B

        Returns
        -------
        numpy.ndarray
            The shape of this vector is (n_atoms + 1,)
        """
        B = np.zeros(self.n_atoms + 1)
        for conformer in self.conformers:
            B[:-1] += conformer.weighted_b_matrix
        B /= self.n_conformers
        B[-1] = self.charge
        return B

    def finalize_geometries(self,
                            executor: Optional[concurrent.futures.Executor] = None,
                            timeout: Optional[float] = None,
                            command_log: str ="finalize_geometry_commands.log"):
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
        utils.run_with_executor(functions, executor=executor, timeout=timeout,
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
        utils.run_with_executor(functions, executor=executor, timeout=timeout,
                                command_log=command_log)

    def run(self,
            executor: Optional[concurrent.futures.Executor] = None,
            timeout: Optional[float] = None,
            geometry_command_log: str = "finalize_geometry_commands.log",
            esp_command_log: str = "finalize_geometry_commands.log",
            ) -> npt.NDArray:
        
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

        self.stage_1_charges = None
        self.stage_2_charges = None

        self.finalize_geometries(executor=executor, timeout=timeout,
                                 command_log=geometry_command_log)
        self.compute_esps(executor=executor, timeout=timeout,
                          command_log=esp_command_log)

        initial_charge_options = ChargeOptions(**self.charge_options.dict())
        stage_1 = RespStageOptions.from_model(self, hyp_a=self.hyp_a1)
        
        if stage_2:
            final_charge_options = ChargeOptions(**initial_charge_options)
            initial_charge_options.charge_equivalences = []
        else:
            final_charge_options = initial_charge_options

        sp3_ch_ids = psi4utils.get_sp3_ch_ids(self.psi4mol)
        final_charge_options.add_sp3_equivalences(sp3_ch_ids)

        a_matrix = self.get_conformer_a_matrix()
        b_matrix = self.get_conformer_b_matrix(executor=executor)

        self.stage_1_charges = RespCharges(symbols=self.symbols,
                                           charge_options=initial_charge_options,
                                           resp_stage_options=stage_1)
        q1 = self.stage_1_charges.fit(a_matrix, b_matrix)

        if self.resp_options.stage_2:
            final_charge_options.add_stage_2_constraints(q1)
            stage_2 = RespStageOptions.from_model(self, hyp_a=self.hyp_a2)
            self.stage_2_charges = RespCharges(symbols=self.symbols,
                                               charge_options=final_charge_options,
                                               resp_stage_options=stage_2)
            self.stage_2_charges.fit(a_matrix, b_matrix)

        return self.charges
