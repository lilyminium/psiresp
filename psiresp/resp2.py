import numpy as np

from .conformer import Conformer
from .resp import Resp
from .multiresp import MultiResp
from . import utils
from .due import due, Doi
from .options import IOOptions, QMOptions, OrientationOptions, ChargeOptions, ESPOptions


@due.dcite(
    Doi("10.1038/s42004-020-0291-4"),
    description="RESP2",
    path="psiresp.resp2",
)
class Resp2:
    """
    Class to manage R/ESP for one molecule of multiple conformers.

    Parameters
    ----------
    conformers: iterable of Conformers
        conformers of the molecule. They must all have the same atoms
        in the same order.
    name: str (optional)
        name of the molecule. This is used to name output files. If not
        given, defaults to 'Mol'.
    charge: int (optional)
        overall charge of the molecule.
    multiplicity: int (optional)
        multiplicity of the molecule
    charge_constraint_options: psiresp.ChargeOptions (optional)
        charge constraints and charge equivalence constraints
    io_options: psiresp.IOOptions (optional)
        input/output options
    delta: float (optional)
            Delta factor used to weight the gas and solvent contributions
            to the final charges.
    
    

    Attributes
    ----------
    name: str
        name of the molecule. This is used to name output files.
    gas: :class:`psiresp.resp.Resp`
        Vacuum Resp class
    solvated: :class:`psiresp.resp.Resp`
        Solvated Resp class
    delta: float
        Delta factor used to weight the gas and solvent contributions
        to the final charges.
    gas_charges: numpy.ndarray
        Final charges of the vacuum system.
        This is populated upon calling ``run()``; otherwise, it is ``None``.
    solvated_charges: numpy.ndarray
        Final charges of the solvated system.
        This is populated upon calling ``run()``; otherwise, it is ``None``.
    charges: numpy.ndarray
        Final charges of the RESP2 system. This is calculated from the
        gas charges and solvated charges, weighted by the ``delta`` factor.
    """

    @classmethod
    def from_molecules(cls,
                       molecules,
                       name="Mol",
                       charge=0,
                       multiplicity=1,
                       io_options=IOOptions(),
                       qm_options=QMOptions(),
                       esp_options=ESPOptions(),
                       orientation_options=OrientationOptions(),
                       charge_constraint_options=ChargeOptions(),
                       weights=None,
                       optimize_geometry=False,
                       delta=0.6):
        """
        Create Resp2 class from Psi4 molecules.

        Parameters
        ----------
        molecules: iterable of Psi4 molecules
            conformers of the molecule. They must all have the same atoms
            in the same order.
        name: str (optional)
            name of the molecule. This is used to name output files. If not
            given, defaults to 'Mol'.
        qm_options: psiresp.QMOptions (optional)
            Psi4 QM job options
        esp_options: psiresp.ESPOptions (optional)
            Options for generating the grid for computing ESP
        orientation_options: psiresp.OrientationOptions (optional)
            Options for generating orientations for each conformer
        charge_constraint_options: psiresp.ChargeOptions (optional)
            charge constraints and charge equivalence constraints
        io_options: psiresp.IOOptions (optional)
            input/output options
        weights: list of float (optional)
            The weights to assign to each molecule conformer
            in the RESP job. Must be of same length as ``molecules``
        optimize_geometry: bool (optional)
            Whether to optimize the geometry of each conformer
        delta: float (optional)
            Delta factor used to weight the gas and solvent contributions
            to the final charges.

        Returns
        -------
        resp: :class:`psiresp.resp2.Resp2`
        """
        molecules = utils.asiterable(molecules)
        # weights = utils.asiterable(weights)
        n_molecules = len(molecules)
        if weights is None:
            weights = np.ones(n_molecules)
        elif len(weights) != n_molecules:
            msg = ("`weights` must be an iterable of values of same length "
                   f"as `molecules`. Cannot assign {len(weights)} weights to "
                   f"{n_molecules} molecules")
            raise ValueError(msg)

        conformers = []
        for i, (mol, weight) in enumerate(zip(molecules, weights), 1):
            cname = f"{name}_c{i:03d}"
            mol.activate_all_fragments()
            mol.set_molecular_charge(charge)
            mol.set_multiplicity(multiplicity)
            conf = Conformer(mol.clone(),
                             name=cname,
                             charge=charge,
                             multiplicity=multiplicity,
                             optimize_geometry=optimize_geometry,
                             weight=weight,
                             io_options=io_options,
                             qm_options=qm_options,
                             esp_options=esp_options,
                             orientation_options=orientation_options)
            conformers.append(conf)
        return cls(conformers,
                   name=name,
                   charge=charge,
                   multiplicity=multiplicity,
                   io_options=io_options,
                   charge_constraint_options=charge_constraint_options,
                   delta=delta)

    def __init__(self,
                 conformers=[],
                 name="Resp",
                 charge=0,
                 multiplicity=1,
                 charge_constraint_options=ChargeOptions(),
                 io_options=IOOptions(),
                 delta=0.6):
        if name is None:
            name = "Resp"
        self.name = name
        base = Resp(conformers, charge=charge, multiplicity=multiplicity,
                    io_options=io_options)
        self.gas = base.clone(name=name + "_gas")
        for mol in self.gas.conformers:
            mol.qm_options.solvent = None
            mol.qm_options.method = "PW6B95"
            mol.qm_options.basis = "aug-cc-pV(D+d)Z"
            mol.esp_options.vdw_point_density = 2.5
            mol.esp_options.use_radii = "bondi"
        self.solvated = self.gas.clone(name=name + "_solvated")
        for mol in self.gas.conformers:
            mol.qm_options.solvent = "water"
        self.delta = delta

    @property
    def charges(self):
        return self.delta * self.solvated.charges + (1 - self.delta) * self.gas.charges

    @property
    def gas_charges(self):
        return self.gas.charges

    @property
    def solvated_charges(self):
        return self.solvated.charges

    def run(self, executor=None, charge_constraint_options=None, tol: float = 1e-6, maxiter: int = 50):

        self.gas.run(executor=executor,
                     stage_2=True,
                     charge_constraint_options=charge_constraint_options,
                     restrained=True,
                     hyp_a1=0.0005,
                     hyp_a2=0.001,
                     hyp_b=0.1,
                     ihfree=True,
                     tol=tol,
                     maxiter=maxiter)

        self.solvated.run(executor=executor,
                          stage_2=True,
                          charge_constraint_options=charge_constraint_options,
                          restrained=True,
                          hyp_a1=0.0005,
                          hyp_a2=0.001,
                          hyp_b=0.1,
                          ihfree=True,
                          tol=tol,
                          maxiter=maxiter)
        return self.charges


@due.dcite(
    Doi("10.1038/s42004-020-0291-4"),
    description="RESP2 multi-molecule fit",
    path="psiresp.resp2",
)
class MultiResp2(object):
    """
    Class to manage Resp2 for multiple molecules of multiple conformers.

    Parameters
    ----------
    resps: list of Resp
        Molecules for multi-molecule fit, set up in Resp classes.
        This should *not* be Resp2 instances.
    charge_constraint_options: psiresp.ChargeOptions (optional)
        Charge constraints and charge equivalence constraints.
        When running a fit, both these *and* the constraints supplied
        in each individual RESP class are taken into account. This is
        to help with differentiating between intra- and inter-molecular
        constraints.

    Attributes
    ----------
    gas: :class:`psiresp.multiresp.MultiResp`
        Vacuum MultiResp class
    solvated: :class:`psiresp.multiresp.MultiResp`
        Solvated psiresp class
    delta: float
        Delta factor used to weight the gas and solvent contributions
        to the final charges.
    gas_charges: list numpy.ndarray
        Final charges of the vacuum system.
        This is populated upon calling ``run()``; otherwise, it is ``None``.
    solvated_charges: list numpy.ndarray
        Final charges of the solvated system.
        This is populated upon calling ``run()``; otherwise, it is ``None``.
    charges: list numpy.ndarray
        Final charges of the RESP2 system. This is calculated from the
        gas charges and solvated charges, weighted by the ``delta`` factor.
    """
    def __init__(self, resps, charge_constraint_options=ChargeOptions(), delta=0.6):
        self.delta = delta
        # in case they've passed Resp2 instances
        base_resps = []
        for r in resps:
            try:
                r = r.gas
            except AttributeError:
                pass
            else:
                r.name = r.name.strip("_gas")
            base_resps.append(r)
        cresps = [r.clone(name=r.name + "_gas") for r in base_resps]
        for resp in cresps:
            for mol in resp.conformers:
                mol.qm_options.solvent = None
                mol.qm_options.method = "PW6B95"
                mol.qm_options.basis = "aug-cc-pV(D+d)Z"
                mol.esp_options.vdw_point_density = 2.5
                mol.esp_options.use_radii = "bondi"
        self.gas = MultiResp(cresps, charge_constraint_options=charge_constraint_options)
        self.solvated = self.gas.clone(suffix="_solvated")
        for resp in cresps:
            for mol in resp.conformers:
                mol.qm_options.solvent = "water"

    def run(self, executor=None, charge_constraint_options=None, tol: float = 1e-6, maxiter: int = 50):

        self.gas.run(executor=executor,
                     stage_2=True,
                     charge_constraint_options=charge_constraint_options,
                     restrained=True,
                     hyp_a1=0.0005,
                     hyp_a2=0.001,
                     hyp_b=0.1,
                     ihfree=True,
                     tol=tol,
                     maxiter=maxiter)

        self.solvated.run(executor=executor,
                          stage_2=True,
                          charge_constraint_options=charge_constraint_options,
                          restrained=True,
                          hyp_a1=0.0005,
                          hyp_a2=0.001,
                          hyp_b=0.1,
                          ihfree=True,
                          tol=tol,
                          maxiter=maxiter)

    @property
    def charges(self):
        charges = []
        for solv, gas in zip(self.solvated_charges, self.gas_charges):
            charges.append(self.delta * solv + (1 - self.delta) * gas)
        return charges

    @property
    def gas_charges(self):
        return self.gas.charges

    @property
    def solvated_charges(self):
        return self.solvated.charges
