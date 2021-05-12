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
class Resp2(object):
    @classmethod
    def from_molecules(cls, molecules, name="Mol", charge=0,
                       multiplicity=1, io_options=IOOptions(),
                       qm_options = QMOptions(),
                       esp_options = ESPOptions(),
                       orientation_options = OrientationOptions(),
                       charge_constraint_options = ChargeOptions(),
                       weights=None, optimize_geometry=False,
                       delta=0.6):
        """
        Create Resp class from Psi4 molecules.

        Parameters
        ----------
        molecules: iterable of Psi4 molecules
            conformers of the molecule. They must all have the same atoms
            in the same order.
        name: str (optional)
            name of the molecule. This is used to name output files. If not
            given, defaults to 'Mol'.
        **kwargs:
            arguments passed to ``Resp.__init__()``.

        Returns
        -------
        resp: Resp
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
            conf = Conformer(mol.clone(), name=cname, charge=charge,
                             multiplicity=multiplicity, optimize_geometry=optimize_geometry,
                             weight=weight, io_options=io_options, qm_options=qm_options,
                             esp_options=esp_options, orientation_options=orientation_options)
            conformers.append(conf)
        return cls(conformers, name=name, charge=charge, multiplicity=multiplicity,
                   io_options=io_options, charge_constraint_options=charge_constraint_options,
                   delta=delta)

    def __init__(self, conformers=[], name="Resp", charge=0,
                 multiplicity=1, charge_constraint_options=ChargeOptions(),
                 io_options=IOOptions(), delta=0.6):
        if name is None:
            name = "Resp"
        self.name = name
        self.gas = Resp(conformers).clone(name=name + "_gas")
        for mol in self.gas.conformers:
            mol.qm_options.solvent = None
            mol.qm_options.method = "PW6B95"
            mol.qm_options.basis = "aug-cc-pV(D+d)Z"
            mol.esp_options.rmin = 1.3
            mol.esp_options.rmax = 2.1
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


    def run(self, executor=None, charge_constraint_options=None, tol: float=1e-6,
            maxiter: int=50):

        self.gas.run(executor=executor, stage_2=True, charge_constraint_options=charge_constraint_options,
                     restrained=True, hyp_a1=0.0005, hyp_a2=0.001, hyp_b=0.1, ihfree=True, tol=tol,
                     maxiter=maxiter)

        self.solvated.run(executor=executor, stage_2=True, charge_constraint_options=charge_constraint_options,
                     restrained=True, hyp_a1=0.0005, hyp_a2=0.001, hyp_b=0.1, ihfree=True, tol=tol,
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

    Attributes
    ----------
    gas: Resp
        MultiResp class of molecules in gaseous phase
    solv: Resp
        MultiResp class of molecules in aqueous phase
    gas_charges: ndarray of floats
        RESP charges in gas phase (only exists after calling run)
    solv_charges: ndarray of floats
        RESP charges in aqueous phase (only exists after calling run)
    charges: ndarray of floats
        Resp2 charges (only exists after calling run)
    """

    def __init__(self, resps, charge_constraint_options=ChargeOptions(), delta=0.6):
        self.delta = delta
        cresps = [r.clone(name=r.name + "_gas") for r in resps]
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

    def run(self, executor=None, charge_constraint_options=None, tol: float=1e-6,
            maxiter: int=50):

        self.gas.run(executor=executor, stage_2=True, charge_constraint_options=charge_constraint_options,
                     restrained=True, hyp_a1=0.0005, hyp_a2=0.001, hyp_b=0.1, ihfree=True, tol=tol,
                     maxiter=maxiter)

        self.solvated.run(executor=executor, stage_2=True, charge_constraint_options=charge_constraint_options,
                     restrained=True, hyp_a1=0.0005, hyp_a2=0.001, hyp_b=0.1, ihfree=True, tol=tol,
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
