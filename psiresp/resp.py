import logging
from dataclasses import dataclass, field
import concurrent.futures

import numpy as np

from .conformer import Conformer
from . import utils, base, psi4utils, exceptions, constants
from .options import ChargeOptions, RespOptions, RespStageOptions
from .charges import RespStageCharges

logger = logging.getLogger(__name__)


@dataclass
class Resp(base.MoleculeBase):

    charge: int = 0
    multiplicity: int = 1
    charge_constraint_options: ChargeOptions = field(default_factory=ChargeOptions)
    resp_options: RespOptions = field(default_factory=RespOptions)
    qm_options: QMOptions = field(default_factory=QMOptions)
    grid_options: GridOptions = field(default_factory=GridOptions)
    conformer_name_template = "{name}_c{counter:03d}"

    _child_class = Conformer

    def __post_init__(self):
        super().__post_init__()
        self.conformers = []
        self.stage_1_charges = None
        self.stage_2_charges = None

    @property
    def _child_container(self):
        return self.conformers

    @property
    def _child_name_template(self):
        return self.conformer_name_template

    @property
    def n_conformers(self):
        return len(self.conformers)

    def add_conformer(self, coordinates_or_psi4mol, **kwargs):
        try:
            xyz = coordinates_or_psi4mol.geometry().np.astype("float")
        except AttributeError:
            pass
        else:
            coordinates_or_psi4mol = xyz * constants.BOHR_TO_ANGSTROM
        return self._add_child(coordinates_or_psi4mol, **kwargs)

    def generate_conformers(self):
        self.conformers = []
        self.conformer_options.generate_conformer_geometries(self.psi4mol)
        for coordinates in self.conformer_options.conformer_geometries:
            self._add_child(coordinates)

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

    def get_conformer_a_matrix(self):
        A = np.zeros((self.n_atoms + 1, self.n_atoms + 1))
        for conformer in self.conformers:
            A[:-1, :-1] += conformer.get_weighted_a_matrix()
        A /= self.n_conformers
        A[-1, :-1] = A[:-1, -1] = 1
        return A

    def get_conformer_b_matrix(self, executor=None):
        B = np.zeros(self.n_atoms + 1)
        for conformer in self.conformers:
            B[:-1] += conformer.get_weighted_b_matrix(executor=executor)
        B /= self.n_conformers
        B[-1] = self.charge
        return B

    def finalize_geometries(self, executor=None, timeout=None,
                            command_log="finalize_geometry_commands.log"):
        functions = [conf.finalize_geometry for conf in self.conformers]
        utils.run_with_executor(functions, executor=executor, timeout=timeout,
                                command_log=command_log)
    
    def compute_esps(self, executor=None, timeout=None,
                     command_log="compute_esp_commands.log"):
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
            ):
        self.stage_1_charges = None
        self.stage_2_charges = None

        self.finalize_geometries(executor=executor, timeout=timeout,
                                 command_log=geometry_command_log)
        self.compute_esps(executor=executor, timeout=timeout,
                          command_log=esp_command_log)

        initial_charge_options = ChargeOptions(**self.charge_options)
        stage_1 = RespStageOptions.from_resp_options(self.resp_options,
                                                     hyp_a_name="hyp_a1")
        
        if stage_2:
            final_charge_options = ChargeOptions(**initial_charge_options)
            initial_charge_options.charge_equivalences = []
        else:
            final_charge_options = initial_charge_options

        sp3_ch_ids = psi4utils.get_sp3_ch_ids(self.psi4mol)
        final_charge_options.add_sp3_equivalences(sp3_ch_ids)

        a_matrix = self.get_conformer_a_matrix()
        b_matrix = self.get_conformer_b_matrix(executor=executor)

        self.stage_1_charges = RespStageCharges(symbols=self.symbols,
                                                **initial_charge_options,
                                                **stage_1)
        q1 = self.stage_1_charges.fit(a_matrix, b_matrix)

        if self.resp_options.stage_2:
            final_charge_options.add_stage_2_constraints(q1)
            stage_2 = RespStageOptions.from_resp_options(self.resp_options,
                                                         hyp_a_name="hyp_a2")
            self.stage_2_charges = RespStageCharges(symbols=self.symbols,
                                                    **final_charge_options,
                                                    **stage_2)
            self.stage_2_charges.fit(a_matrix, b_matrix)

        return self.charges
