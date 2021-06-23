import logging

import numpy as np

from .conformer import Conformer
from . import utils, base
from .options import (ChargeOptions, RespOptions, RespCharges, IOOptions, QMOptions, ESPOptions, OrientationOptions)

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
    charge: int (optional)
        overall charge of the molecule.
    multiplicity: int (optional)
        multiplicity of the molecule
    charge_constraint_options: psiresp.ChargeOptions (optional)
        charge constraints and charge equivalence constraints
    io_options: psiresp.IOOptions (optional)
        input/output options
    
    

    Attributes
    ----------
    conformers: list of Conformers
    indices: ndarray of ints
        indices of each atom, indexed from 0
    name: str
        name of the molecule. This is used to name output files.
    charge: int
        overall charge of the molecule.
    multiplicity: int
        multiplicity of the molecule
    n_atoms: int
        number of atoms in each conformer
    symbols: ndarray
        element symbols
    charge_constraint_options: psiresp.ChargeOptions
        charge constraints and charge equivalence constraints
    stage_1_charges: psiresp.RespCharges
        This is populated upon calling ``run()``; otherwise, it is ``None``.
    stage_2_charges: psiresp.RespCharges
        This is populated upon calling ``run()`` with ``stage_2=True``.
        Otherwise, it is ``None``.
    charges: numpy.ndarray of floats
        These are the final charges of the molecule. It is only populated
        upon calling ``run()``. Otherwise, it is ``None``.

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
                       optimize_geometry=False):
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

        Returns
        -------
        resp: :class:`psiresp.resp.Resp`
        """
        molecules = utils.asiterable(molecules)
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
                   charge_constraint_options=charge_constraint_options)

    @classmethod
    def from_rdmol(cls, rdmol, name="Mol", rmsd_threshold=1.5, minimize=False,
                   n_confs=0, minimize_maxIters=2000, **kwargs):
        """Generate conformers from an RDKit molecule and create a RESP
        instance from them

        Parameters
        ----------
        rdmol: rdkit.Chem.Mol
        name: str (optional)
            name of the molecule.
        rmsd_threshold: float (optional)
            RMSD threshold to generate different conformers
        minimize: bool (optional)
            Whether to minimize each generated conformer
        n_confs: int (optional)
            number of conformers to generate. If 0, no limit is applied
        minimize_maxIters: int (optional)
            maximum number of iterations to minimize conformers with, if
            ``minimize=True``
        **kwargs:
            passed to Resp.from_molecules
        
        Returns
        -------
        resp: Resp
        """
        from rdkit import Chem
        from rdkit.Chem import AllChem
        rdmol = Chem.AddHs(rdmol)

        AllChem.EmbedMultipleConfs(rdmol, numConfs=n_confs,
                                   pruneRmsThresh=rmsd_threshold,
                                   useRandomCoords=True,
                                   ignoreSmoothingFailures=True)
        if minimize:
            # TODO: is UFF good?
            AllChem.UFFOptimizeMoleculeConfs(rdmol,
                                             maxIters=minimize_maxIters)
        molecules = utils.rdmol_to_psi4mols(rdmol, name=name)

        return cls.from_molecules(molecules, name=name, **kwargs)

    def __init__(self,
                 conformers=[],
                 name="Resp",
                 charge=0,
                 multiplicity=1,
                 charge_constraint_options=ChargeOptions(),
                 io_options=IOOptions()):
        super(Resp, self).__init__(name=name, io_options=io_options)
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

        log.debug(f"Resp(name={self.name}) created with " f"{len(self.conformers)} conformers")

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
        """Create a MDAnalysis.Universe from first conformer
        
        Returns
        -------
        MDAnalysis.Universe
        """
        u = self.conformers[0].to_mda()
        if self.charges is not None:
            u.add_TopologyAttr("charges", self.charges)
        return u

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
        new = type(self)(conformers,
                         name=name,
                         charge=self.charge,
                         multiplicity=self.multiplicity,
                         io_options=self.io_options,
                         charge_constraint_options=self.charge_constraint_options)
        return new


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

    def fit(self, resp_options=RespOptions(), charge_constraint_options=None,
            executor=None):
        a_matrix = self.get_conformer_a_matrix()
        b_matrix = self.get_conformer_b_matrix(executor=executor)

        if charge_constraint_options is None:
            charge_constraint_options = self.charge_constraint_options

        A, B = charge_constraint_options.get_constraint_matrix(a_matrix, b_matrix)

        resp_options = RespOptions(**resp_options)
        resp_charges = RespCharges(resp_options, symbols=self.symbols,
                                   n_structures=self.n_structure_array)
        resp_charges.fit(A, B)
        return resp_charges

    def compute_optimized_geometry(self, executor=None):
        for conformer in self.conformers:
            conformer.compute_optimized_geometry(executor=executor)

    def _run(self,
             executor=None,
             stage_1_options=RespOptions(hyp_a=0.0005),
             stage_2_options=RespOptions(hyp_a=0.001),
             stage_2=False,
             charge_constraint_options=None):

        self.compute_optimized_geometry(executor=executor)

        if charge_constraint_options is None:
            charge_constraint_options = self.charge_constraint_options

        initial_charge_options = ChargeOptions(**charge_constraint_options)

        if stage_2:
            final_charge_options = ChargeOptions(**initial_charge_options)
            # final_charge_options.charge_constraints = []
            initial_charge_options.charge_equivalences = []
        else:
            final_charge_options = initial_charge_options

        if initial_charge_options.equivalent_methyls:
            final_charge_options.add_methyl_equivalences(self.sp3_ch_ids)

        a_matrix = self.get_conformer_a_matrix()
        b_matrix = self.get_conformer_b_matrix(executor=executor)
        print(initial_charge_options)

        a1, b1 = initial_charge_options.get_constraint_matrix(a_matrix, b_matrix)
        stage_1_options = RespOptions(**stage_1_options)
        self.stage_1_charges = RespCharges(stage_1_options,
                                           symbols=self.symbols,
                                           n_structures=self.n_structure_array)
        self.stage_1_charges.fit(a1, b1)

        if stage_2:
            final_charge_options.add_stage_2_constraints(self.stage_1_charges.charges,
                                                         sp3_ch_ids=self.sp3_ch_ids)
            print(final_charge_options)

            a2, b2 = final_charge_options.get_constraint_matrix(a_matrix, b_matrix)
            self.stage_2_charges = RespCharges(stage_2_options,
                                               symbols=self.symbols,
                                               n_structures=self.n_structure_array)
            self.stage_2_charges.fit(a2, b2)

        return self.charges

    def run(self,
            executor=None,
            stage_2: bool = False,
            charge_constraint_options=None,
            restrained: bool = True,
            hyp_a1: float = 0.0005,
            hyp_a2: float = 0.001,
            hyp_b: float = 0.1,
            ihfree: bool = True,
            tol: float = 1e-6,
            maxiter: int = 50):
        """
        Runs charge calculation based on given parameters. This populates the
        ``stage_1_charges`` and, optionally, ``stage_2_charges`` attributes.

        Parameters
        ----------
        executor: concurrent.futures.Executor (optional)
            Executor used to run parallel QM jobs. If not provided, the code
            runs in serial.
        stage_2: bool (optional)
            Whether to run a two stage fit
        charge_constraint_options: psiresp.ChargeOptions (optional)
            Charge constraint options to use while fitting the charges. If not
            provided, the options stored in the ``charge_constraint_options``
            attribute are used. Providing this argument does not store the options
            in the attribute.
        restrained: bool (optional)
            Whether to perform a restrained fit
        hyp_a1: float (optional)
            scale factor of asymptote limits of hyperbola, in the stage 1 fit
        hyp_a2: float (optional)
            scale factor of asymptote limits of hyperbola, in the stage 2 fit
        hyp_b: float (optional)
            tightness of hyperbola at its minimum
        ihfree: bool (optional)
            if True, exclude hydrogens from restraint
        tol: float (optional)
            threshold for convergence
        maxiter: int (optional)
            maximum number of iterations to solve constraint matrices

        Returns
        -------
        numpy.ndarray: charges
            The final charges
        """

        stage_1_options = RespOptions(restrained=restrained,
                                      hyp_a=hyp_a1,
                                      hyp_b=hyp_b,
                                      ihfree=ihfree,
                                      tol=tol,
                                      maxiter=maxiter)
        stage_2_options = RespOptions(restrained=restrained,
                                      hyp_a=hyp_a2,
                                      hyp_b=hyp_b,
                                      ihfree=ihfree,
                                      tol=tol,
                                      maxiter=maxiter)
        return self._run(stage_1_options=stage_1_options,
                         stage_2_options=stage_2_options,
                         charge_constraint_options=charge_constraint_options,
                         executor=executor,
                         stage_2=stage_2)
