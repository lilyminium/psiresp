

import pathlib
from typing import List, Dict, Optional, Union
import yaml

import numpy as np
import scipy
import psi4
from pydantic import Field

from .mixins import RespMixin, RespCharges, RespMoleculeOptions, ChargeConstraintOptions, IOMixin
from .resp import Resp
from .utils import psi4utils


class MultiResp(RespMixin, IOMixin):
    """
    Class to manage R/ESP for multiple molecules of multiple conformers.

    Parameters
    ----------
    charge_constraint_options: psiresp.ChargeConstraintOptions (optional)
        Charge constraints and charge equivalence constraints.
        When running a fit, both these *and* the constraints supplied
        in each individual RESP class are taken into account. This is
        to help with differentiating between intra- and inter-molecular
        constraints.

    Attributes
    ----------
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
    name: str = Field(
        default="multiresp",
        description=("Name for the job. "
                     "This affects which directory to save files to."),
    )
    resps: List[Resp] = Field(
        default_factory=list,
        description="Resp classes for multi-molecule fit",
    )
    molecule_options: RespMoleculeOptions = Field(
        default_factory=RespMoleculeOptions,
        description="Options for creating new Resp instances",
    )

    @classmethod
    def from_yaml(cls, filename):
        with open(filename, "r") as f:
            content = yaml.full_load(f)

        molecules = content.pop("molecules", {})
        obj = cls(**content)
        global_mol_options = obj.molecule_options.to_kwargs()

        if not molecules:
            molecules[obj.name] = {}
            obj.name = "multiresp"

        for name, local_mol_options in molecules.items():
            options = dict(**global_mol_options)
            options["name"] = name
            options.update(local_mol_options)
            name = options["name"]
            try:
                molfile = options.pop("molfile")
            except KeyError:
                raise TypeError("a `molfile` must be given for each molecule "
                                "containing the molecule specification. "
                                "A `molfile` was not given for the molecule "
                                f"named '{name}'. "
                                "Accepted formats include PDB, XYZ, MOL2.")
            else:
                molfile = molfile.format(**options)
            resp = Resp.from_molfile(molfile, **options)
            obj.add_resp(resp)
        return obj

    def __init__(self, *args, resps=[], **kwargs):
        if args and len(args) == 1 and not resps:
            resps = args[0]
            super().__init__(**kwargs)
        else:
            super().__init__(*args, **kwargs)
        for resp in resps:
            self.add_resp(resp)  # lgtm [py/init-calls-subclass]
            resp.parent = self

    def _set_charges(self, charges, stage=1):
        super()._set_charges(charges, stage=stage)
        i = 0
        n_orientations = self.n_orientation_array
        for resp in self.resps:
            j = i + resp.n_atoms
            resp_charges = RespCharges(symbols=resp.symbols,
                                       n_orientations=n_orientations[i:j])
            resp_charges._start_index = 0
            resp_charges._charge_object = charges
            resp._set_charges(resp_charges, stage=stage)

    @property
    def path(self):
        if self.directory_path is not None:
            return self.directory_path
        return pathlib.Path(self.name)

    def generate_conformers(self):
        for resp in self.resps:
            resp.generate_conformers()

    @property
    def resps_by_name(self):
        return {resp.name: resp for resp in self.resps}

    @property
    def conformers(self):
        for resp in self.resps:
            for conformer in resp.conformers:
                yield conformer

    @property
    def n_conformers(self):
        return sum(resp.n_conformers for resp in self.resps)

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
            default_kwargs = self.molecule_options.to_kwargs()
            # TODO: this is a bad hack
            # TODO: automatically set molecule number at this part?
            default_kwargs["charge_constraint_options"] = {}
            default_kwargs.update(kwargs)
            psi4mol_or_resp = Resp.from_model(self, psi4mol=mol, name=name, **default_kwargs)

        psi4mol_or_resp.parent = self
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
        sparse = scipy.sparse.block_diag(matrices).tocsr()
        return sparse  # / self.n_conformers

    def get_conformer_b_matrix(self):
        """Assemble the conformer B matrices of each Resp molecule

        Returns
        -------
        numpy.ndarray
            The shape of this array is (n_total_atoms,)
        """
        matrices = [resp.get_conformer_b_matrix() for resp in self.resps]
        return np.concatenate(matrices)  # / self.n_conformers

    @property
    def n_orientation_array(self):
        structures = []
        for resp in self.resps:
            structures.extend([resp.n_orientations] * resp.n_atoms)
        return structures

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
        sparse = scipy.sparse.bmat(inputs).tocsr()
        return sparse

    def get_b_matrix(self):
        """Assemble the B matrices of each Resp molecule

        Returns
        -------
        numpy.ndarray
            The shape of this array is (n_total_atoms + n_resps,)
        """
        b = self.get_conformer_b_matrix()
        charges = [resp.charge for resp in self.resps]
        matrix = np.concatenate([b, charges])
        return matrix

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
            opts = mol.charge_constraint_options.copy(deep=True)
            ignore = []
            for constraint in opts.iterate_over_constraints():
                if constraint.some_molecule_ids_defined():
                    raise ValueError("All molecule IDs must be defined or None. "
                                     "A mix of values is not accepted. Given: "
                                     f"{constraint.molecule_ids}")
                if not constraint.any_molecule_ids_defined():
                    constraint.set_molecule_ids(i)
                    constraint.set_atom_increment(mapping[i])

                elif len(set(constraint.molecule_ids)) == 1:
                    if constraint.molecule_ids[0] == i:
                        constraint.set_atom_increment(mapping[i])
                else:
                    ignore.append(constraint)

            equivalences = [eq for eq in opts.charge_equivalences
                            if eq not in ignore]
            constraints = [con for con in opts.charge_constraints
                           if con not in ignore]
            options.charge_equivalences.extend(equivalences)
            options.charge_constraints.extend(constraints)
        options.clean_charge_constraints()
        options.clean_charge_equivalences()
        return options

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
