
from typing import Optional
from pydantic import PrivateAttr, Field

# from .mixins import RespMoleculeOptions, MoleculeMixin, RespMixin
from .resp import Resp
from .utils import psi4utils
from .utils.due import due, Doi


@due.dcite(
    Doi("10.1038/s42004-020-0291-4"),
    description="RESP2",
    path="psiresp.resp2",
)
class Resp2(Resp):
    """Class to manage one Resp2 job"""
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
        self.qm_options._gas_phase_by_name = True
        self.qm_options._gas_phase_name = "_gas"

        gas_name = f"{self.name}_gas"
        self._gas_phase = Resp.from_model(self, name=gas_name)
        self._gas_phase.qm_options = self.qm_options
        self._gas_phase.grid_options = self.grid_options
        self._gas_phase.directory_path = self.path / gas_name

        solv_name = f"{self.name}_solvated"
        self._solvated_phase = Resp.from_model(self, name=solv_name)
        self._solvated_phase.qm_options = self.qm_options
        self._solvated_phase.grid_options = self.grid_options
        self._solvated_phase.directory_path = self.path / solv_name

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
        self.gas.add_conformer(mol, name=name, **kwargs)
        self.solvated.add_conformer(mol, name=name, **kwargs)

    @property
    def conformers(self):
        return self.gas.conformers + self.solvated.conformers

    def _fit_resp_charges(self):
        self.gas._fit_resp_charges()
        self.solvated._fit_resp_charges()
