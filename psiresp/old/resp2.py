
from typing import Optional, Union
from pydantic import PrivateAttr, Field
import concurrent.futures

import psi4

# from .mixins import RespMoleculeOptions, MoleculeMixin, RespMixin
from .resp import Resp
from .multiresp import MultiResp
from .utils import psi4utils
from .utils.due import due, Doi


@due.dcite(
    Doi("10.1038/s42004-020-0291-4"),
    description="RESP2",
    path="psiresp.resp2",
)
class Resp2Mixin:
    """Class to manage one Resp2 job"""

    @property
    def gas(self):
        return self._gas_phase

    @property
    def solvated(self):
        return self._solvated_phase

    
    def _assign_qm_grid_options(self):
        self.qm_options._gas_phase_by_name = True
        self.qm_options._gas_phase_name = "_gas"

        for phase in self.phases:
            name = phase.name
            phase.qm_options = self.qm_options
            phase.grid_options = self.grid_options
            phase.directory_path = self.path / name

    @property
    def phases(self):
        return [self.gas, self.solvated]


@due.dcite(
    Doi("10.1038/s42004-020-0291-4"),
    description="RESP2",
    path="psiresp.resp2",
)
class Resp2(Resp2Mixin, Resp):
    # grid_rmin: float = 1.3
    # grid_rmax: float = 2.1
    solvent: str = "water"
    qm_method: str = "pw6b95"
    qm_basis_set: str = "aug-cc-pV(D+d)Z"
    vdw_point_density: float = 2.5
    use_radii: str = "bondi"
    delta: float = Field(
        default=0.6,
        description=("Weight that controls how much the solvated "
                     "charges contribute to the final charges. The "
                     "closer this is to 1, the higher the contribution. "
                     "Conversely, the contribution from gas phase "
                     "charges lowers as delta -> 1."),
    )

    _gas_phase: Resp = PrivateAttr()
    _solvated_phase: Resp = PrivateAttr()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fix_charge_and_multiplicity()
        gas_name = f"{self.name}_gas"
        self._gas_phase = Resp.from_model(self, name=gas_name)
        solv_name = f"{self.name}_solvated"
        self._solvated_phase = Resp.from_model(self, name=solv_name)
        self._assign_qm_grid_options()

    def __setstate__(self, state):
        super().__setstate__(state)
        self._assign_qm_grid_options()
        self._pin_conformer_coordinates(self, self._gas_phase)
        self._pin_conformer_coordinates(self, self._solvated_phase)

    def add_conformer(self,
                      coordinates_or_psi4mol: psi4utils.CoordinateInputs,
                      name: Optional[str] = None,
                      **kwargs):
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
            Arguments used to construct the Conformer.
            If not provided, the default specification given in
            :attr:`psiresp.resp.Resp.conformer_options`
            will be used.
        """
        mol = psi4utils.psi4mol_with_coordinates(self.psi4mol,
                                                 coordinates_or_psi4mol)
        super().add_conformer(mol, name=name, **kwargs)
        self.gas.add_conformer(mol, name=name, **kwargs)
        self.solvated.add_conformer(mol, name=name, **kwargs)

    @property
    def conformers(self):
        # TODO: maybe I should not do this?
        return self.gas.conformers + self.solvated.conformers

    @property
    def charges(self):
        return self.delta * self.solvated.charges + (1 - self.delta) * self.gas.charges

    def _fit_resp_charges(self):
        for phase in self.phases:
            phase._fit_resp_charges()


@due.dcite(
    Doi("10.1038/s42004-020-0291-4"),
    description="RESP2",
    path="psiresp.resp2",
)
class MultiResp2(Resp2Mixin, MultiResp):
    name: str = "multiresp2"
    # grid_rmin: float = 1.3
    # grid_rmax: float = 2.1
    solvent: str = "water"
    qm_method: str = "pw6b95"
    qm_basis_set: str = "aug-cc-pV(D+d)Z"
    vdw_point_density: float = 2.5
    use_radii: str = "bondi"
    delta: float = Field(
        default=0.6,
        description=("Weight that controls how much the solvated "
                     "charges contribute to the final charges. The "
                     "closer this is to 1, the higher the contribution. "
                     "Conversely, the contribution from gas phase "
                     "charges lowers as delta -> 1."),
    )

    _gas_phase: MultiResp = PrivateAttr()
    _solvated_phase: MultiResp = PrivateAttr()

    def __init__(self, *args, resps=[], **kwargs):
        if args and len(args) == 1 and not resps:
            resps = args[0]
            super().__init__(**kwargs)
        else:
            super().__init__(*args, **kwargs)
        self._gas_phase = MultiResp.from_model(self, name="")
        self._solvated_phase = MultiResp.from_model(self, name="")
        self._assign_qm_grid_options()

        for resp in resps:
            self.add_resp(resp)
            resp.parent = self

    def __setstate__(self, state):
        super().__setstate__(state)
        self._assign_qm_grid_options()

    @property
    def charges(self):
        return self.delta * self.solvated.charges + (1 - self.delta) * self.gas.charges

    def _fit_resp_charges(self):
        for phase in self.phases:
            phase._fit_resp_charges()

    def add_resp(self,
                 psi4mol_or_resp: Union[psi4.core.Molecule, Resp],
                 name: Optional[str] = None,
                 **kwargs) -> Resp:
        """Add Resp, possibly creating from Psi4 molecule

        Parameters
        ----------
        psi4mol_or_resp: psi4.core.Molecule or Resp
            Psi4 Molecule or Resp instance. If this is a molecule,
            the molecule is copied before creating the Resp. If it is
            a Resp instance, the Resp is just appended to
            :attr:`psiresp.multiresp.MultiResp.resps`.
        name: str (optional)
            Name of Resp. If not provided, one will be generated automatically
        **kwargs:
            Arguments used to construct the Resp.
            If not provided, the default specification given in
            :attr:`psiresp.multiresp.MultiResp`
            will be used.

        Returns
        -------
        resp: Resp
        """
        super().add_resp(psi4mol_or_resp, name=name, **kwargs)
        resp = self.resps[-1]
        gas_phase = Resp.from_model(resp, name=f"{resp.name}_gas")
        self.gas.add_resp(gas_phase)

        solv_phase = Resp.from_model(resp, name=f"{resp.name}_solvated")
        self.solvated.add_resp(solv_phase)

        for phase in self.phases:
            last_resp = phase.resps[-1]
            last_resp.conformers = []
            for conf in resp.conformers:
                last_resp.add_conformer(conf.psi4mol)
                last_conf = last_resp.conformers[-1]
                last_conf.qm_options = self.qm_options
                last_conf.grid_options = self.grid_options
                for orientation in conf.orientations:
                    last_conf.add_orientation(orientation.psi4mol)
                    last_orientation = last_conf.orientations[-1]
                    last_orientation.grid_options = self.grid_options
                    last_orientation.qm_options = self.qm_options
            # phase.resps[-1]._conformer_coordinates = resp._conformer_coordinates
            
            # for conformer in phase.resps[-1].conformers:
            #     conformer.qm_options = self.qm_options
            #     conformer.grid_options = self.grid_options
            #     for orientation in conformer.orientations:
            #         orientation.qm_options = self.qm_options
            #         orientation.grid_options = self.grid_options

    def generate_conformers(self):
        for i, resp in enumerate(self.resps):
            resp.generate_conformers()
            for phase in self.phases:
                phase_resp = phase.resps[i]
                phase_resp._conformer_coordinates = resp._conformer_coordinates
                phase_resp.conformers = []
                phase_resp.generate_conformers()

    @property
    def conformers(self):
        for phase in [self.gas, self.solvated]:
            for resp in phase.resps:
                for conformer in resp.conformers:
                    yield conformer

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
        # TODO: necessary? Or can I rely on the RespMixin?
        self.finalize_geometries(executor=executor, timeout=timeout)
        for orient in self.orientations:
            orient.compute_grid(grid_options=self.grid_options)
        functions = [orient.compute_esp for orient in self.orientations]
        self.qm_options.run_with_executor(functions, executor=executor, timeout=timeout,
                                          command_log=command_log)
