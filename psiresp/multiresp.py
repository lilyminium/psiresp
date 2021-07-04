from typing import List, Dict, Optional, Union

import numpy as np
import scipy
import psi4

from .mixins import RespMixin, RespMoleculeOptions, ChargeConstraintOptions
from .resp import Resp


class MultiResp(RespMixin):
    """
    Class to manage R/ESP for multiple molecules of multiple conformers.

    Parameters
    ----------
    resps: list of Resp
        Molecules for multi-molecule fit, set up in Resp classes.
    charge_constraint_options: psiresp.ChargeConstraintOptions (optional)
        Charge constraints and charge equivalence constraints.
        When running a fit, both these *and* the constraints supplied
        in each individual RESP class are taken into account. This is
        to help with differentiating between intra- and inter-molecular
        constraints.

    Attributes
    ----------
    molecules: list of Resp
        Molecules for multi-molecule fit, set up in Resp classes.
    n_molecules: int
        number of molecule Resp classes
    n_structures: int
        number of structures in entire MultiResp fit, where one structure
        is one orientation of one conformer
    n_atoms: int
        total number of atoms in the fit (sum of all atoms in each Resp class)
    symbols: ndarray
        all the element symbols in the fit
    charges: list of ndarray
        partial atomic charges for each molecule
        (only exists after calling run)
    """

    resps: List[Resp] = []

    def __post_init__(self):
        super().__post_init__()
        for resp in self.resps:
            resp.resp = self

    @property
    def resps_by_name(self):
        return {resp.name: resp for resp in self.resps}

    @property
    def conformers(self):
        for resp in self.resps:
            for conformer in resp.conformers:
                yield conformer

    @property
    def symbols(self):
        values = []
        for resp in self.resps:
            values.extend(resp.symbols)
        return values

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
        if not isinstance(psi4mol_or_resp, Resp):
            mol = psi4mol_or_resp.clone()
            if name is None:
                name = f"Mol_{len(self.resps) + 1:03d}"
            psi4mol_or_resp = Resp.from_model(self, psi4mol=mol, name=name)

        psi4mol_or_resp.resp = self
        self.resps.append(psi4mol_or_resp)
        return psi4mol_or_resp

    def copy(self, suffix="_copy"):
        """Copy into another instance of MultiResp

        Parameters
        ----------
        suffix: str (optional)
            This is appended to each of the names of the molecule Resps
            in the MultiResp

        Returns
        -------
        MultiResp
        """
        names = [r.name + suffix for r in self.resps]
        resps = [m.copy(name=n) for n, m in zip(names, self.resps)]
        kwargs = self.dict()
        kwargs["resps"] = resps
        return type(self)(**kwargs)

    @property
    def n_resps(self):
        return len(self.resps)

    def get_conformer_a_matrix(self):
        """Assemble the conformer A matrices of each Resp molecule

        Returns
        -------
        numpy.ndarray
            The shape of this array is (n_total_atoms, n_total_atoms)
        """
        matrices = [scipy.sparse.coo_matrix(resp.get_conformer_a_matrix())
                    for resp in self.resps]
        return scipy.sparse.block_diag(matrices)

    def get_conformer_b_matrix(self):
        """Assemble the conformer B matrices of each Resp molecule

        Returns
        -------
        numpy.ndarray
            The shape of this array is (n_total_atoms,)
        """
        matrices = [resp.get_conformer_b_matrix() for resp in self.resps]
        return np.concatenate(matrices)

    def get_a_matrix(self):
        """Assemble the A matrices of each Resp molecule

        Returns
        -------
        numpy.ndarray
            The shape of this array is
            (n_total_atoms + n_resps, n_total_atoms + n_resps)
        """
        a = self.get_conformer_a_matrix()
        matrices = []
        for resp in self.resps:
            arr = np.ones(resp.n_atoms)
            matrices.append(scipy.sparse.coo_matrix(arr))
        rows = scipy.sparse.block_diag(matrices)
        inputs = [[a, rows.T], [rows, None]]
        return scipy.sparse.bmat(inputs)

    def get_b_matrix(self):
        """Assemble the B matrices of each Resp molecule

        Returns
        -------
        numpy.ndarray
            The shape of this array is (n_total_atoms + n_resps,)
        """
        b = self.get_conformer_b_matrix()
        charges = [resp.charge for resp in self.charges]
        return np.concatenate([b, charges])

    @property
    def resp_atom_increments(self):
        n_atoms = [resp.n_atoms for resp in self.resps]
        edges = np.cumsum(np.r_[0, (*n_atoms,)])[:-1]
        return {i: e for i, e in enumerate(edges, 1)}

    def get_clean_charge_options(self) -> ChargeConstraintOptions:
        """Get clean charge constraints from MultiResp.

        This runs over each Resp and adds the correct atom increment
        to each constraint.

        Returns
        -------
        options: ChargeConstraintOptions
        """
        mapping = self.resp_atom_increments
        options = self.charge_constraint_options.copy(deep=True)
        # add atom increments to atoms
        for constraint in options.iterate_over_constraints():
            for atom_id in constraint.atom_ids:
                if atom_id.molecule_id is None:
                    raise ValueError("Molecule IDs should be specified for "
                                     "all multimolecular fits")
                atom_id.atom_increment = mapping[atom_id.molecule_id]
        # incorporate intramolecular constraints
        # n_atoms = 0
        for i, mol in enumerate(self.resps, 1):
            opts = ChargeConstraintOptions(**mol.charge_constraint_options)
            ignore = []
            for constraint in opts.iterate_over_constraints():
                if constraint.some_molecule_ids_defined():
                    raise ValueError("All molecule IDs must be defined or None. "
                                     "A mix of values is not accepted. Given: "
                                     f"{constraint.molecule_ids}")
                if not constraint.any_molecule_ids_defined():
                    constraint.molecule_ids = i
                    constraint.atom_increments = mapping[i]

                elif len(set(constraint.molecule_ids)) == 1:
                    if constraint.molecule_ids[0] == i:
                        constraint.atom_increments = mapping[i]
                else:
                    ignore.append(constraint)

            equivalences = [eq for eq in opts.charge_equivalences
                            if eq not in ignore]
            constraints = [con for con in opts.charge_constraints
                           if con not in ignore]
            multiopts.charge_equivalences.extend(equivalences)
            multiopts.charge_constraints.extend(constraints)
        multiopts.clean_charge_constraints()
        multiopts.clean_charge_equivalences()
        return multiopts

    def get_sp3_ch_ids(self) -> Dict[int, List[int]]:
        """Get dictionary of sp3 carbon atom number to bonded hydrogen numbers.

        These atom numbers are indexed from 1. Each key is the number of an
        sp3 carbon. The value is the list of bonded hydrogen numbers.

        Returns
        -------
        c_h_dict: dict of {int: list of ints}
        """
        sp3_ch_ids = {}
        i = 0
        for resp in self.resps:
            resp_ids = psi4utils.get_sp3_ch_ids(resp.psi4mol, increment=i)
            sp3_ch_ids.update(resp_ids)
            i += resp.n_atoms
        return sp3_ch_ids
