import functools
from typing import List, Dict, Set, Tuple
from collections import defaultdict

import qcelemental as qcel
import scipy.sparse
import numpy as np
from pydantic import Field, validator

from psiresp import base, psi4utils
from psiresp.molecule import Atom, Molecule
from psiresp.constraint import SparseGlobalConstraintMatrix


@functools.total_ordering
class BaseChargeConstraint(base.Model):
    atoms: Set[Atom] = set()

    @classmethod
    def from_molecule(cls, molecule, indices=[], **kwargs):
        atoms = [Atom(molecule=molecule, index=i) for i in indices]
        return cls(atoms=atoms, **kwargs)

    def __len__(self):
        return len(self.indices)

    def __hash__(self):
        return hash(frozenset(self.atoms))

    def __eq__(self, other):
        if type(other) != type(self):
            return False
        return self.atoms == other.atoms

    def __lt__(self, other):
        return sorted(self.atoms)[0] < sorted(other.atoms)[0]

    @property
    def indices(self):
        return np.array([a.index for a in self.atoms])

    @property
    def molecules(self):
        return np.array([a.molecule for a in self.atoms])

    @property
    def molecule_set(self):
        return {a.molecule for a in self.atoms}

    def to_sparse_row_constraint(self, n_dim: int,
                                 molecule_increments: Dict[int, int] = {}):
        raise NotImplementedError

    # @functools.lru_cache
    def to_sparse_col_constraint(self, n_dim: int,
                                 molecule_increments: Dict[int, int] = {}):
        return self.to_sparse_row_constraint(n_dim, molecule_increments).transpose()

    def get_atom_indices(self, molecule_increments: Dict[int, int] = {}):
        indices = [atom.index + molecule_increments.get(hash(atom.molecule), 0)
                   for atom in self.atoms]
        return np.array(sorted(indices), dtype=int)


class ChargeSumConstraint(BaseChargeConstraint):
    charge: float = Field(default=0,
                          description="Specified charge")

    def __eq__(self, other):
        return super().__eq__(other) and self.charge == other.charge

    # @functools.lru_cache
    # def to_sparse_row_constraint(self, n_dim: int,
    #                              molecule_increments: Dict[Molecule, int] = {}):
    #     n_items = len(self)
    #     ones = np.ones(n_items)
    #     row = np.zeros(n_items, dtype=int)
    #     indices = self.get_atom_indices(molecule_increments)

    #     return scipy.sparse.coo_matrix(
    #         (ones, (row, indices)),
    #         shape=(1, n_dim)
    #     )

    def to_row_constraint(self, n_dim: int,
                          molecule_increments: Dict[Molecule, int] = {}):
        row = np.zeros((1, n_dim))
        indices = self.get_atom_indices(molecule_increments)
        row[0, indices] = 1
        return row


    def __hash__(self):
        return hash((frozenset(self.atoms), self.charge))


class ChargeEquivalenceConstraint(BaseChargeConstraint):

    # @functools.lru_cache
    # def to_sparse_row_constraint(self, n_dim: int,
    #                              molecule_increments: Dict[Molecule, int] = {}):
    #     n_items = len(self) - 1
    #     ones = np.ones(n_items)
    #     row = np.tile(np.arange(n_items, dtype=int), 2)
    #     indices = self.get_atom_indices(molecule_increments)
    #     col = np.concatenate([indices[:-1], indices[1:]])

    #     return scipy.sparse.coo_matrix(
    #         (np.concatenate([-ones, ones]), (row, col)),
    #         shape=(n_items, n_dim)
    #     )

    def to_row_constraint(self, n_dim: int,
                          molecule_increments: Dict[Molecule, int] = {}):
        n_items = len(self) - 1
        rows = np.zeros((n_items, n_dim))
        indices = self.get_atom_indices(molecule_increments)
        for i, (j, k) in enumerate(zip(indices[:-1], indices[1:])):
            rows[i][j] = -1
            rows[i][k] = 1
        return rows

    @property
    def charge(self):
        return None


class BaseChargeConstraintOptions(base.Model):
    charge_sum_constraints: List[ChargeSumConstraint] = []
    charge_equivalence_constraints: List[ChargeEquivalenceConstraint] = []

    @property
    def n_constraints(self):
        return (len(self.charge_sum_constraints)
                + len(self.charge_equivalence_constraints))
    
    def iter_constraints(self):
        yield from self.charge_sum_constraints
        yield from self.charge_equivalence_constraints

    def add_charge_sum_constraint(self, charge, atoms=[]):
        self.charge_sum_constraints.append(ChargeSumConstraint(charge=charge, atoms=atoms))

    def add_charge_equivalence_constraint(self, atoms=[]):
        self.charge_equivalence_constraints.append(ChargeEquivalenceConstraint(atoms=atoms))

    def add_charge_sum_constraint_for_molecule(self, molecule, charge=0, indices=[]):
        atoms = Atom.from_molecule(molecule=molecule, indices=indices)
        return self.add_charge_sum_constraint(charge=charge, atoms=atoms)

    def add_charge_equivalence_constraint_for_molecule(self, molecule, indices=[]):
        atoms = Atom.from_molecule(molecule=molecule, indices=indices)
        return self.add_charge_equivalence_constraint(atoms=atoms)

    def _unite_overlapping_equivalences(self):
        """Join ChargeEquivalenceConstraints with overlapping atoms"""
        equivalences = defaultdict(set)
        for chrequiv in self.charge_equivalence_constraints:
            for atom in chrequiv.atoms:
                equivalences[atom] |= chrequiv.atoms

        self.charge_equivalence_constraints = [ChargeEquivalenceConstraint(atoms=x)
                                               for x in equivalences.values()]

    def _get_single_atom_charge_constraints(self) -> Dict[Atom, float]:
        """Get ChargeConstraints with only one atom as a dict"""
        single_charges = {}
        for constr in self.charge_sum_constraints:
            if len(constr.atoms) == 1:
                atom = list(constr.atoms)[0]
                if atom in single_charges:
                    err = ("Found conflicting charge constraints for "
                           f"atom {atom}, constrained to both "
                           f"{single_charges[atom]} and {constr.charge}")
                    raise ValueError(err)
                single_charges[atom] = constr.charge
        return single_charges

    def _remove_incompatible_and_redundant_equivalent_atoms(self):
        """Remove atoms from charge equivalences if they are constrained
        to different charges, and remove charge equivalence constraints
        if all atoms are constrained to the same charge (so it is redundant)
        """
        # if a charge equivalence has multiple atoms constrained to
        # single, different, charges, remove those.
        single_charges = self._get_single_atom_charge_constraints()
        redundant = []
        for i, constraint in enumerate(self.charge_equivalence_constraints):
            single_atoms = [(atom, single_charges[atom])
                            for atom in constraint.atoms
                            if atom in single_charges]
            if len(single_atoms):
                atoms, charges = zip(*single_atoms)
                if len(set(charges)) > 1:
                    # TODO: silently delete or raise an error?
                    constraint.atoms -= set(atoms)
                elif len(charges) == len(constraint.atoms):
                    # every atom in the equivalence is constrained to the same charge
                    # this is redundant and should get removed
                    # or can result in singular matrices
                    redundant.append(i)

        for i in redundant[::-1]:
            del self.charge_equivalence_constraints[i]

    def _remove_redundant_charge_constraints(self):
        single_charges = self._get_single_atom_charge_constraints()
        redundant = []
        for i, constraint in enumerate(self.charge_sum_constraints):
            if len(constraint.atoms) > 1:
                if all(atom in single_charges for atom in constraint.atoms):
                    redundant.append(i)
        for i in redundant[::-1]:
            del self.charge_sum_constraints[i]

    def clean_charge_equivalence_constraints(self):
        """Clean the ChargeEquivalence constraints.

        1. Join charge equivalence constraints with overlapping atoms
        2. Remove atoms from charge equivalences if they are constrained
        to different charges, and remove charge equivalence constraints
        if all atoms are constrained to the same charge (so it is redundant)
        """
        self._unite_overlapping_equivalences()
        self._remove_incompatible_and_redundant_equivalent_atoms()
        constraint_set = set(self.charge_equivalence_constraints)
        self.charge_equivalence_constraints = sorted(constraint_set)

    def clean_charge_sum_constraints(self):
        self._remove_redundant_charge_constraints()
        self.charge_sum_constraints = sorted(set(self.charge_sum_constraints))


class ChargeConstraintOptions(BaseChargeConstraintOptions):
    symmetric_methyls: bool = True
    symmetric_methylenes: bool = True




class MoleculeChargeConstraints(BaseChargeConstraintOptions):
    molecules: List[Molecule] = []
    unconstrained_atoms: List[Atom] = []

    _n_atoms: int
    _n_molecule_atoms: np.ndarray
    _molecule_increments: Dict[int, int]
    _edges: List[Tuple[int, int]]



    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        n_atoms = [mol.n_atoms for mol in self.molecules]
        self._n_atoms = sum(n_atoms)
        self._n_molecule_atoms = np.cumsum(np.r_[0, n_atoms])
        self._molecule_increments = {}
        for mol, i in zip(self.molecules, self._n_molecule_atoms):
            self._molecule_increments[hash(mol)] = i #dict(zip(self.molecules, self._n_molecule_atoms))
        self._edges = list(zip(self._n_molecule_atoms[:-1],
                               self._n_molecule_atoms[1:]))

    @property
    def n_atoms(self):
        return self._n_atoms

    @classmethod
    def from_charge_constraints(cls, charge_constraints, molecules=[]):
        molecule_set = set(molecules)
        sums = [constr.copy(deep=True)
                for constr in charge_constraints.charge_sum_constraints
                if constr.molecule_set & molecule_set]
        eqvs = [constr.copy(deep=True)
                for constr in charge_constraints.charge_equivalence_constraints
                if constr.molecule_set & molecule_set]

        constraints = cls(charge_sum_constraints=sums,
                          charge_equivalence_constraints=eqvs,
                          molecules=molecules)

        accepted = []
        if charge_constraints.symmetric_methyls:
            accepted.append(3)
        if charge_constraints.symmetric_methylenes:
            accepted.append(2)
        if accepted:
            constraints.add_sp3_equivalences(accepted)
        return constraints

    def to_a_col_constraints(self):
        return [constr.to_row_constraint(n_dim=self._n_atoms + len(self.molecules),
                                         molecule_increments=self._molecule_increments).T
                for constr in self.iter_constraints()]
    
    def to_b_constraints(self):
        b = [constr.charge for constr in self.charge_sum_constraints]
        return np.array(b)

    # def construct_sparse_column_block(self):
    #     n_atoms = sum(m.qcmol.geometry.shape[0] for m in self.molecules)
    #     increments = self.get_molecule_increments()

    #     columns = [c.to_sparse_col_constraint(n_atoms, increments)
    #                for c in self.constraints()]
    #     col_block = scipy.sparse.hstack(columns, format="coo")
    #     return col_block

    # def construct_global_constraint_matrix(self, surface_constraints,
    #                                        mask=None):
    #     a = scipy.sparse.csr_matrix(surface_constraints.a)
    #     b = surface_constraints.b

    #     if self.n_constraints:
    #         a_block_col, b_block = self._generate_charge_constraint_column()
    #         a = scipy.sparse.bmat(
    #             [[a, a_block_col],
    #              [a_block_col.transpose(), None]]
    #         )
    #         b = np.r_[b, b_block]
    #     return SparseGlobalConstraintMatrix(a, b, self._n_atoms, mask=mask)

    # def _generate_charge_constraint_column(self):
    #     n_atoms = sum(m.qcmol.geometry.shape[0] for m in self.molecules) #+ 1
    #     increments = self.get_molecule_increments()

    #     constraints = [*self.charge_sum_constraints,
    #                    *self.charge_equivalence_constraints]
    #     col_ = [c.to_sparse_col_constraint(n_atoms, increments)
    #             for c in constraints]
    #     a_block = scipy.sparse.hstack(col_, format="csr")
    #     b_charges = [c.charge for c in self.charge_sum_constraints]
    #     b_block = np.zeros(a_block.shape[1])
    #     b_block[:len(b_charges)] = b_charges
    #     return a_block, b_block

    # def to_col_constraints(self):
    #     increments = self.get_molecule_increments()
    #     for constr in self.constraints:
    #         yield constr.to_row_constraint(n_dim=self.n_atoms,
    #                                        molecule_increments=increments).T



    def add_constraints_from_charges(self, charges):
        unconstrained_atoms = [atom
                               for eqv in self.charge_equivalence_constraints
                               for atom in eqv.atoms]
        unconstrained_atoms += self.unconstrained_atoms

        mol = list(self._molecule_increments.keys())[0]
        unconstrained_indices = [a.index + self._molecule_increments[hash(a.molecule)]
                                 for a in unconstrained_atoms]

        indices = np.arange(self.n_atoms)
        to_constrain = np.where(~np.in1d(indices, unconstrained_indices))[0]
        charges = np.asarray(charges)[to_constrain]
        indices = indices[to_constrain]

        for i, q in zip(indices, charges):
            self.add_charge_sum_constraint_from_indices(charge=q, indices=[i])

        self.clean_charge_sum_constraints()
        self.clean_charge_equivalence_constraints()


    def add_charge_sum_constraint_from_indices(self, charge, indices=[]):
        atoms = [self._atom_from_index(i) for i in indices]
        constraint = ChargeSumConstraint(charge=charge,
                                         atoms=atoms)
        self.charge_sum_constraints.append(constraint)

    def _atom_from_index(self, index):
        i = np.searchsorted(self._n_molecule_atoms[1:], index, side="left")
        return Atom(molecule=self.molecules[i],
                    index=index - self._n_molecule_atoms[i])

    def _index_array(self, array):
        return [array[i:j] for i, j in self._edges]

    def add_sp3_equivalences(self, accepted_n_hs=(2, 3)):
        """
        """
        for mol in self.molecules:
            ch_groups = psi4utils.get_sp3_ch_indices(mol.qcmol)
            for c, hs in ch_groups.items():
                if len(hs) in accepted_n_hs:
                    atoms = [Atom(molecule=mol, index=i) for i in hs]
                    self.charge_equivalence_constraints.append(
                        ChargeEquivalenceConstraint(atoms=atoms)
                    )
                    self.unconstrained_atoms.append(
                        Atom(molecule=mol, index=c)
                    )
