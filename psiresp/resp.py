import logging
import io

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import MDAnalysis as mda

from .conformer import Conformer
from . import utils, base
from .options import (ChargeOptions, RespOptions, RespCharges, IOOptions,
                      QMOptions, ESPOptions, OrientationOptions)

log = logging.getLogger(__name__)


class Resp(base.IOBase):
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
    chrconstr: list or dict (optional)
        Intramolecular charge constraints in the form of
        {charge: atom_number_list} or [(charge, atom_number_list)].
        The numbers are indexed from 1. e.g. {0: [1, 2]} or [[0, [1, 2]]]
        mean that atoms 1 and 2 together have a charge of 0.
    chrequiv: list (optional)
        lists of atoms with equivalent charges, indexed from 1.
        e.g. [[1, 2], [3, 4, 5]] mean that atoms 1 and 2 have equal
        charges, and atoms 3, 4, and 5 have equal charges.

    Attributes
    ----------
    name: str
        name of the molecule. This is used to name output files.
    n_atoms: int
        number of atoms in each conformer
    indices: ndarray of ints
        indices of each atom, indexed from 0
    heavy_ids: ndarray of ints
        atom numbers of heavy atoms, indexed from 1
    h_ids: ndarray of ints
        atom numbers of hydrogen atoms, indexed from 1
    symbols: ndarray
        element symbols
    charge: int or float
        overall charge of molecule
    conformers: list of Conformers
    n_conf: int
        number of conformers
    unrestrained_charges: ndarray of floats
        partial atomic charges (only exists after calling fit or run)
    restrained_charges: ndarray of floats
        partial atomic charges (only exists after calling fit or run
        with restraint=True)
    """

    # @classmethod
    # def from_molecule_string(cls, molecules, name="Mol", executor=None, **kwargs):
    #     """
    #     Create Resp class from Psi4 molecules.

    #     Parameters
    #     ----------
    #     molecules: iterable of Psi4 molecules
    #         conformers of the molecule. They must all have the same atoms
    #         in the same order.
    #     name: str (optional)
    #         name of the molecule. This is used to name output files. If not
    #         given, defaults to 'Mol'.
    #     **kwargs:
    #         arguments passed to ``Resp.__init__()``.

    #     Returns
    #     -------
    #     resp: Resp
    #     """
    #     conformers = []
    #     if executor is None:
    #         executor = concurrent.futures.ThreadPoolExecutor()
    #     for i, mol in enumerate(molecules, 1):
    #         cname = f"{name}_c{i:03d}"
    #         conformers.append(executor.submit(Conformer, mol, name=cname, **kwargs))
    #     confs = [c.result() for c in conformers]
    #     return cls(confs, name=name, executor=executor, **kwargs)

    @classmethod
    def from_molecules(cls, molecules, name="Mol", charge=0,
                       multiplicity=1, io_options=IOOptions(),
                       qm_options = QMOptions(),
                       esp_options = ESPOptions(),
                       orientation_options = OrientationOptions(),
                       charge_constraint_options = ChargeOptions(),
                       weights=None, optimize_geometry=False):
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
                   io_options=io_options, charge_constraint_options=charge_constraint_options)

    # @classmethod
    # def from_rdmol(cls, rdmol, name="Mol", rmsd_threshold=1.5, minimize=False,
    #                n_confs=0, minimize_maxIters=2000, **kwargs):
    #     rdmol = Chem.AddHs(rdmol)
    #     confs = []

    #     cids = AllChem.EmbedMultipleConfs(rdmol,
    #                                       numConfs=n_confs,
    #                                       pruneRmsThresh=rmsd_threshold,
    #                                       ignoreSmoothingFailures=True)

    #     if minimize:
    #         # TODO: is UFF good?
    #         AllChem.UFFOptimizeMoleculeConfs(rdmol,
    #                                          maxIters=minimize_maxIters)
    #     molecules = utils.rdmol_to_psi4mols(rdmol, name=name)

    #     return cls.from_molecules(molecules, name=name, **kwargs)

    def __init__(self, conformers=[], name="Resp", charge=0,
                 multiplicity=1, charge_constraint_options=ChargeOptions(),
                 io_options=IOOptions()):
        super().__init__(name=name, io_options=io_options)
        if not conformers:
            raise ValueError("Resp must be created with at least one conformer")
        self.conformers = utils.asiterable(conformers)

        # # molecule information
        conf = conformers[0]
        self.charge = charge
        self.multiplicity = multiplicity
        self.symbols = conf.symbols
        self.n_atoms = conf.n_atoms
        self.sp3_ch_ids = utils.get_sp3_ch_ids(conf.psi4mol)

        # mol info
        self.indices = np.arange(self.n_atoms)
        self.charge_constraint_options = charge_constraint_options

        self.stage_1_charges = None
        self.stage_2_charges = None

        log.debug(f"Resp(name={self.name}) created with "
                  f"{len(self.conformers)} conformers")

    @property
    def charge(self):
        return self.conformers[0].charge

    @charge.setter
    def charge(self, value):
        for conf in self.conformers:
            conf.charge = value

    @property
    def multiplicity(self):
        return self.conformers[0].multiplicity

    @multiplicity.setter
    def multiplicity(self, value):
        for conf in self.conformers:
            conf.multiplicity = value

    @property
    def charge_constraint_options(self):
        return self._charge_constraint_options

    @charge_constraint_options.setter
    def charge_constraint_options(self, options):
        self._charge_constraint_options = cc = ChargeOptions(**options)
        if cc.equivalent_methyls:
            cc.add_methyl_equivalences(self.sp3_ch_ids)
    
    @property
    def resp_stage_options(self):
        return self._resp_stage_options

    @resp_stage_options.setter
    def resp_stage_options(self, option_list):
        options = [RespOptions(**x) for x in option_list]
        self._resp_stage_options = options


    @property
    def charges(self):
        if self.stage_2_charges is not None:
            return self.stage_2_charges.charges
        try:
            return self.stage_1_charges.charges
        except AttributeError:
            return self.stage_1_charges

    @property
    def n_structures(self):
        return sum([c.n_orientations for c in self.conformers])
    
    @property
    def n_structure_array(self):
        return np.repeat(self.n_structures, self.n_atoms)

    def to_mda(self):
        mol = utils.psi42xyz(self.conformers[0].molecule)
        u = mda.Universe(io.StringIO(mol), format="XYZ")
        u.add_TopologyAttr("charges", self.charges)
        return u

    def write(self, filename):
        u = self.to_mda()
        u.atoms.write(filename)

    def clone(self, name=None):
        """Clone into another instance of Resp

        Parameters
        ----------
        name: str (optional)
            If not given, the new Resp instance has '_copy' appended
            to its name

        Returns
        -------
        resp: Resp
        """
        if name is None:
            name = self.name + "_copy"
        conformers = [c.clone(name=f"{name}_c{i:03d}") for i, c in enumerate(self.conformers, 1)]
        new = type(self)(conformers, name=name, charge=self.charge,
                        multiplicity=self.multiplicity, io_options=self.io_options,
                        charge_constraint_options=self.charge_constraint_options)
        return new

    # # def optimize_geometry(self, psi4_options={}):
    # #     """
    # #     Optimise geometry for all conformers.

    # #     Parameters
    # #     ----------
    # #     basis: str (optional)
    # #         Basis set to optimise geometry
    # #     method: str (optional)
    # #         Method to optimise geometry
    # #     psi4_options: dict (optional)
    # #         additional Psi4 options
    # #     save_opt_geometry: bool (optional)
    # #         if ``True``, saves optimised geometries to XYZ file
    # #     save_files: bool (optional)
    # #         if ``True``, Psi4 files are saved. This does not affect
    # #         writing optimised geometries to files.
    # #     """
    # #     # self._client.map(lambda x: x.optimize_geometry, self.conformers, psi4_options=psi4_options)
    # #     for conf in self.conformers:
    # #     #     self._client.submit(conf.optimize_geometry, psi4_options=psi4_options)
    # #         conf.optimize_geometry(psi4_options=psi4_options)

    def get_conformer_a_matrix(self):
        A = np.zeros((self.n_atoms + 1, self.n_atoms + 1))
        for conformer in self.conformers:
            A[:-1, :-1] += conformer.get_weighted_a_matrix()
        A[-1, :-1] = A[:-1, -1] = 1
        return A

    def get_conformer_b_matrix(self, executor=None):
        B = np.zeros(self.n_atoms + 1)
        for conformer in self.conformers:
            B[:-1] += conformer.get_weighted_b_matrix(executor=executor)
        B[-1] = self.charge
        return B


    def fit(self, resp_options=RespOptions(),
            charge_constraint_options=None,
            executor=None):
        
        a_matrix = self.get_conformer_a_matrix()
        b_matrix = self.get_conformer_b_matrix(executor=executor)

        if charge_constraint_options is None:
            charge_constraint_options = self.charge_constraint_options

        A, B = charge_constraint_options.get_constraint_matrix(a_matrix,
                                                               b_matrix)

        resp_options = RespOptions(**resp_options)
        resp_charges = RespCharges(resp_options, symbols=self.symbols,
                                   n_structures=self.n_structure_array)
        resp_charges.fit(A, B)
        return resp_charges

    def compute_optimized_geometry(self, executor=None):
        for conformer in self.conformers:
            conformer.compute_optimized_geometry(executor=executor)

    def _run(self, executor=None, stage_1_options=RespOptions(hyp_a=0.0005),
            stage_2_options=RespOptions(hyp_a=0.001), stage_2=False,
            charge_constraint_options=None):

        self.compute_optimized_geometry(executor=executor)

        if charge_constraint_options is None:
            charge_constraint_options = self.charge_constraint_options

        initial_charge_options = ChargeOptions(**charge_constraint_options)

        if stage_2:
            final_charge_options = ChargeOptions(**initial_charge_options)
            initial_charge_options.charge_equivalences = []
        else:
            final_charge_options = initial_charge_options

        if initial_charge_options.equivalent_methyls:
            final_charge_options.add_methyl_equivalences(self.sp3_ch_ids)
        
        a_matrix = self.get_conformer_a_matrix()
        b_matrix = self.get_conformer_b_matrix(executor=executor)

        a1, b1 = initial_charge_options.get_constraint_matrix(a_matrix, b_matrix)
        stage_1_options = RespOptions(**stage_1_options)
        self.stage_1_charges = RespCharges(stage_1_options, symbols=self.symbols,
                                           n_structures=self.n_structure_array)
        self.stage_1_charges.fit(a1, b1)

        if stage_2:
            final_charge_options.add_stage_2_constraints(self.stage_1_charges.charges,
                                                         sp3_ch_ids=self.sp3_ch_ids)

            a2, b2 = final_charge_options.get_constraint_matrix(a_matrix, b_matrix)
            print(final_charge_options.charge_constraints)
            print(final_charge_options.charge_equivalences)
            self.stage_2_charges = RespCharges(stage_2_options, symbols=self.symbols,
                                            n_structures=self.n_structure_array)
            self.stage_2_charges.fit(a2, b2)

        return self.charges

    def run(self, executor=None, stage_2=False,
            charge_constraint_options=None,
            restrained: bool=True, hyp_a1: float=0.0005,
            hyp_a2=0.001, hyp_b: float=0.1, ihfree: bool=True, tol: float=1e-6,
            maxiter: int=50):
        
        stage_1_options = RespOptions(restrained=restrained, hyp_a=hyp_a1,
                                      hyp_b=hyp_b, ihfree=ihfree, tol=tol,
                                      maxiter=maxiter)
        stage_2_options = RespOptions(restrained=restrained, hyp_a=hyp_a2,
                                      hyp_b=hyp_b, ihfree=ihfree, tol=tol,
                                      maxiter=maxiter)
        return self._run(stage_1_options=stage_1_options,
                         stage_2_options=stage_2_options,
                         charge_constraint_options=charge_constraint_options,
                         executor=executor, stage_2=stage_2)