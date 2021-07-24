
from typing import Optional, Union
from pydantic import PrivateAttr, Field

import psi4

# from .mixins import RespMoleculeOptions, MoleculeMixin, RespMixin
from .resp import Resp
from .mixins import RespMixin
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

    @property
    def charges(self):
        return self.delta * self.solvated.charges + (1 - self.delta) * self.gas.charges

    def _assign_qm_grid_options(self):
        self.qm_options._gas_phase_by_name = True
        self.qm_options._gas_phase_name = "_gas"

        for phase in self.phases:
            name = phase.name
            phase.qm_options = self.qm_options
            phase.grid_options = self.grid_options
            phase.directory_path = self.path / name

    def _fit_resp_charges(self):
        for phase in self.phases:
            phase._fit_resp_charges()

    @property
    def phases(self):
        return [self.gas, self.solvated]


@due.dcite(
    Doi("10.1038/s42004-020-0291-4"),
    description="RESP2",
    path="psiresp.resp2",
)
class Resp2(Resp2Mixin, Resp):
    grid_rmin: float = 1.3
    grid_rmax: float = 2.1
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
        return self.gas.conformers + self.solvated.conformers


@due.dcite(
    Doi("10.1038/s42004-020-0291-4"),
    description="RESP2",
    path="psiresp.resp2",
)
class MultiResp2(Resp2Mixin, MultiResp):
    name: str = "multiresp2"
    grid_rmin: float = 1.3
    grid_rmax: float = 2.1
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
        gas_phase._conformer_coordinates = resp._conformer_coordinates
        self.gas.add_resp(gas_phase)

        solv_phase = Resp.from_model(resp, name=f"{resp.name}_solvated")
        solv_phase._conformer_coordinates = resp._conformer_coordinates
        self.solvated.add_resp(solv_phase)

    def generate_conformers(self):
        for i, resp in enumerate(self.resps):
            resp.generate_conformers()
            for phase in self.phases:
                phase.resps[i]._conformer_coordinates = resp._conformer_coordinates
                phase.generate_conformers()

    @property
    def conformers(self):
        for phase in [self.gas, self.solvated]:
            for resp in phase.resps:
                for conformer in resp.conformers:
                    yield conformer
