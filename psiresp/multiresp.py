import logging

import numpy as np

from .options import ChargeOptions, RespCharges, RespOptions

log = logging.getLogger(__name__)


class MultiResp:
    """
    Class to manage R/ESP for multiple molecules of multiple conformers.

    Parameters
    ----------
    resps: list of Resp
        Molecules for multi-molecule fit, set up in Resp classes.

    Attributes
    ----------
    molecules: list of Resp
        Molecules for multi-molecule fit, set up in Resp classes.
    n_molecules: int
        number of molecule Resp classes
    n_structures: int
        number of structures in entire MultiResp fit, where one structure
        is one orientation of one conformer
    charges: list of ndarray
        partial atomic charges for each molecule
        (only exists after calling fit or run)
    """

    def __init__(self, resps, charge_constraint_options=ChargeOptions()):
        self.molecules = resps
        self._moldct = {r.name: i + 1 for i, r in enumerate(resps)}
        self.symbols = np.concatenate([mol.symbols for mol in self.molecules])
        self.n_atoms = len(self.symbols)
        n_atoms = 0
        self.atom_increment_mapping = {}
        self.sp3_ch_ids = {}
        for i, mol in enumerate(self.molecules, 1):
            self.atom_increment_mapping[i] = n_atoms
            self.atom_increment_mapping[mol.name] = n_atoms
            for c, hs in mol.sp3_ch_ids.items():
                hs = [h + n_atoms for h in hs]
                self.sp3_ch_ids[c + n_atoms] = hs

            n_atoms += mol.n_atoms
        


        self.charge_constraint_options = ChargeOptions(**charge_constraint_options)

        names = ", ".join([m.name for m in self.molecules])
        log.debug(f"Created MultiResp with {self.n_molecules} molecules: {names}")

    def clone(self, suffix="_copy"):
        """Clone into another instance of MultiResp

        Parameters
        ----------
        suffix: str (optional)
            This is appended to each of the names of the molecule Resps
            in the MultiResp

        Returns
        -------
        MultiResp
        """
        names = [r.name + suffix for r in self.molecules]
        resps = [m.clone(name=n) for n, m in zip(names, self.molecules)]
        return type(self)(resps)

    @property
    def n_structures(self):
        return sum([mol.n_structures for mol in self.molecules])

    @property
    def n_structure_array(self):
        n_atoms = [x.n_atoms for x in self.molecules]
        n_structures = [x.n_structures for x in self.molecules]
        return np.repeat(n_structures, n_atoms)

    @property
    def n_molecules(self):
        return len(self.molecules)

    @property
    def charges(self):
        return [mol.charges for mol in self.molecules]

    def get_conformer_a_matrix(self):
        ndim = self.n_atoms + self.n_molecules
        A = np.zeros((ndim, ndim))
        start = 0
        for i, mol in enumerate(self.molecules, 1):
            end = start + mol.n_atoms
            a = mol.get_conformer_a_matrix()
            A[start:end, start:end] = a[:mol.n_atoms, :mol.n_atoms]
            A[-i, start:end+1] = a[-1]
            A[start:end+1, -i] = a[:, -1]
            start = end
        return A

    def get_conformer_b_matrix(self, executor=None):
        B = np.zeros(self.n_atoms + self.n_molecules)
        start = 0
        for i, mol in enumerate(self.molecules, 1):
            end = start + mol.n_atoms
            b = mol.get_conformer_b_matrix(executor=None)
            B[start:end] = b[:-1]
            B[-i] = b[-1]
            start = end
        return B

    def get_absolute_charge_constraint_options(self, charge_constraint_options):
        multiopts = ChargeOptions(**charge_constraint_options)

        # add atom increments to atoms
        for constr in multiopts.iterate_over_constraints():
            for aid in constr.data:
                aid.atom_increment = self.atom_increment_mapping[aid.molecule_id]
        # incorporate intramolecular constraints
        # n_atoms = 0
        for i, mol in enumerate(self.molecules, 1):
            opts = ChargeOptions(**mol.charge_constraint_options)
            for constr in opts.iterate_over_constraints():
                for aid in constr.data:
                    aid.atom_increment = self.atom_increment_mapping[i]
                    aid.molecule_id = i
            multiopts.charge_equivalences.extend(opts.charge_equivalences)
            multiopts.charge_constraints.extend(opts.charge_constraints)
        multiopts.clean_charge_constraints()
        multiopts.clean_charge_equivalences()
        return multiopts

    def _run(self, executor=None, stage_1_options=RespOptions(hyp_a=0.0005),
            stage_2_options=RespOptions(hyp_a=0.001), stage_2=False,
            charge_constraint_options=None):

        for mol in self.molecules:
            mol.compute_optimized_geometry(executor=executor)

        if charge_constraint_options is None:
            charge_constraint_options = self.charge_constraint_options

        initial_charge_options = self.get_absolute_charge_constraint_options(charge_constraint_options)

        if stage_2:
            final_charge_options = ChargeOptions(**initial_charge_options)
            final_charge_options.charge_constraints = []
            initial_charge_options.charge_equivalences = []
        else:
            final_charge_options = initial_charge_options

        if initial_charge_options.equivalent_methyls:
            final_charge_options.add_methyl_equivalences(self.sp3_ch_ids)        
        a_matrix = self.get_conformer_a_matrix()
        b_matrix = self.get_conformer_b_matrix(executor=executor)


        a1, b1 = initial_charge_options.get_constraint_matrix(a_matrix, b_matrix)

        np.savetxt("ac_matrix.dat", a1, fmt="%8.4f")
        np.savetxt("bc_matrix.dat", b1, fmt="%8.4f")

        stage_1_options = RespOptions(**stage_1_options)
        self.stage_1_charges = RespCharges(stage_1_options, symbols=self.symbols,
                                           n_structures=self.n_structure_array)
        self.stage_1_charges.fit(a1, b1)

        for i, mol in enumerate(self.molecules, 1):
            a = self.atom_increment_mapping[i]
            b = a + mol.n_atoms
            mol.stage_1_charges = self.stage_1_charges.copy(start_index=a, end_index=b,
                                                            n_structures=mol.n_structure_array)

        if stage_2:
            final_charge_options.add_stage_2_constraints(self.stage_1_charges.charges,
                                                         sp3_ch_ids=self.sp3_ch_ids)

            a2, b2 = final_charge_options.get_constraint_matrix(a_matrix, b_matrix)
            self.stage_2_charges = RespCharges(stage_2_options, symbols=self.symbols,
                                            n_structures=self.n_structure_array)
            self.stage_2_charges.fit(a2, b2)

            for i, mol in enumerate(self.molecules, 1):
                a = self.atom_increment_mapping[i]
                b = a + mol.n_atoms
                mol.stage_2_charges = self.stage_2_charges.copy(start_index=a, end_index=b,
                                                                n_structures=mol.n_structure_array)


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
