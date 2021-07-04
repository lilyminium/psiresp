from typing import Optional
import concurrent.futures

from pydantic import PrivateAttr

from .mixins import RespMoleculeOptions, IOMixin, MoleculeMixin, RespMixin
from .resp import Resp
from .due import due, Doi


@due.dcite(
    Doi("10.1038/s42004-020-0291-4"),
    description="RESP2",
    path="psiresp.resp2",
)
class Resp2(RespMoleculeOptions, RespMixin, MoleculeMixin):
    """Class to manage one Resp2 job"""
    solvent: Optional[str] = "water"
    qm_method: str = "PW6B95"
    qm_basis_set: str = "aug-cc-pV(D+d)Z"
    vdw_point_density: float = 2.5
    use_radii: str = "bondi"
    delta: float = 0.6

    _gas_phase: Optional[Resp] = PrivateAttr(default=None)
    _solvated_phase: Optional[Resp] = PrivateAttr(default=None)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fix_charge_and_multiplicity()
        self._gas_phase = Resp.from_model(self, name=f"{self.name}_gas",
                                          solvent=None)
        self._solvated_phase = Resp.from_model(self, name=f"{self.name}_solvated")

    @property
    def gas(self):
        return self._gas_phase

    @property
    def solvated(self):
        return self._solvated_phase

    @property
    def charges(self):
        return self.delta * self.solvated.charges + (1 - self.delta) * self.gas.charges

    @property
    def gas_charges(self):
        return self.gas.charges

    @property
    def solvated_charges(self):
        return self.solvated.charges

    def generate_conformers(self):
        self.gas.generate_conformers()
        self.solvated.generate_conformers()

    @property
    def conformers(self):
        for conformer in self.gas.conformers:
            yield conformer
        for conformer in self.solvated.conformers:
            yield conformer

    def run(self,
            executor: Optional[concurrent.futures.Executor] = None,
            timeout: Optional[float] = None,
            geometry_command_log: str = "finalize_geometry_commands.log",
            esp_command_log: str = "finalize_geometry_commands.log",
            ) -> np.ndarray:

        self.generate_conformers()
        self.finalize_geometries(executor=executor, timeout=timeout,
                                 command_log=geometry_command_log)
        self.generate_orientations()
        self.compute_esps(executor=executor, timeout=timeout,
                          command_log=esp_command_log)
        self.gas.run()
        self.solvated.run()
