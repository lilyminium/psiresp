import functools
from typing import List, Dict, Set

import qcelemental as qcel

import scipy
import numpy as np
from pydantic import Field, validator

from . import base, psi4utils
from .molecule import Atom


class BaseChargeConstraint(base.Model):
    atoms: Set[Atom] = set()

    @classmethod
    def from_molecule(cls, molecule, indices=[], **kwargs):
        atoms = [Atom(molecule=molecule, atom_index=i) for i in indices]
        return cls(atoms=atoms, **kwargs)

    def __len__(self):
        return len(self.indices)

    def __hash__(self):
        return hash(frozenset(self.atoms))

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
                                 molecule_increments: Dict[Molecule, int] = {}):
        raise NotImplementedError

    @functools.lru_cache
    def to_sparse_col_constraint(self, n_dim: int,
                                 molecule_increments: Dict[Molecule, int] = {}):
        return self.to_coo_row_constraint(n_dim, molecule_increments).transpose()

    def get_atom_indices(self, molecule_increments: Dict[Molecule, int] = {}):
        return np.array([atom.atom_index + molecule_increments.get(atom.molecule, 0)
                         for atom in self.atoms],
                        dtype=int)


class ChargeSumConstraint(BaseChargeConstraint):
    charge: float = Field(default=0,
                          description="Specified charge")

    @functools.lru_cache
    def to_sparse_row_constraint(self, n_dim: int,
                                 molecule_increments: Dict[Molecule, int] = {}):
        n_items = len(self)
        ones = np.ones(n_items)
        row = np.zeros(n_items, dtype=int)
        indices = self.get_atom_indices(molecule_increments)

        return scipy.sparse.coo_matrix(
            (ones, (row, indices)),
            shape=(n_items, n_dim)
        )

    def __hash__(self):
        return hash((frozenset(self.atoms), self.charge))


class ChargeEquivalenceConstraint(BaseChargeConstraint):

    @functools.lru_cache
    def to_sparse_row_constraint(self, n_dim: int,
                                 molecule_increments: Dict[Molecule, int] = {}):
        n_items = len(self) - 1
        ones = np.ones(n_items)
        row = np.tile(np.arange(n_items, dtype=int), 2)
        indices = self.get_atom_indices(molecule_increments)
        col = np.concatenate(indices[:-1], indices[1:])

        return scipy.sparse.coo_matrix(
            (np.concatenate([-ones, ones]), (row, col)),
            shape=(n_items, n_dim)
        )


class BaseChargeConstraintOptions(base.Model):
    charge_sum_constraints: List[ChargeSumConstraint]
    charge_equivalence_constraints: List[ChargeEquivalenceConstraint]

    def _unite_overlapping_equivalences(self):
        """Join ChargeEquivalenceConstraints with overlapping atoms"""
        equivalences = defaultdict(set)
        for chrequiv in self.charge_equivalence_constraints:
            for atom in atom_set:
                equivalences[atom] |= chrequiv.atoms

        self.charge_equivalence_constraints = [ChargeEquivalenceConstraint(x)
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
            atoms, charges = zip(*[(atom, single_charges[atom])
                                   for atom in constraint.atoms
                                   if atom in single_charges])
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
        self.charge_equivalence_constraints = list(set(self.charge_equivalence_constraints))
        self._unite_overlapping_equivalences()
        self._remove_incompatible_and_redundant_equivalent_atoms()

    def clean_charge_sum_constraints(self):
        self.charge_sum_constraints = list(set(self.charge_sum_constraints))
        self._remove_redundant_charge_constraints()


class ChargeConstraintOptions(BaseChargeConstraintOptions):
    symmetric_methyls: bool = True
    symmetric_methylenes: bool = True


class MoleculeChargeConstraints(BaseChargeConstraintOptions):
    molecules: Tuple[Molecule]
    unconstrained_atoms: List[Atom] = []

    @ classmethod
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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._n_atoms = sum(mol.n_atoms for mol in self.molecules)
        self._n_molecule_atoms = np.cumsum(np.r_[0, self._n_atoms])
        self._molecule_increments = dict(zip(self.molecules, self._n_molecule_atoms))
        self._edges = list(zip(self._n_molecule_atoms[:-1],
                               self._n_molecule_atoms[1:]))

    def get_molecule_increments(self):
        n_atoms = [len(m.qcmol.atomic_numbers) for m in self.molecules]
        n_molecule_atoms = np.r_[0, np.cumsum(n_atoms)]
        return {mol: i for mol, i in zip(self.molecules, n_molecule_atoms)}

    def construct_sparse_column_block(self):
        n_atoms = sum(m.qcmol.geometry.shape[0] for m in self.molecules)
        increments = self.get_molecule_increments()

        columns = [c.to_sparse_col_constraint(n_atoms, increments)
                   for c in self.constraints()]
        col_block = scipy.sparse.hstack(columns, format="coo")
        return col_block

    def construct_constraint_matrix(self):
        n_atoms = sum(m.qcmol.geometry.shape[0] for m in self.molecules)
        increments = self.get_molecule_increments()

        charge_constraint_col = scipy.sparse.hstack(
            [c.to_sparse_col_constraints(n_atoms, increments)
             for c in (*self.charge_sum_constraints,
             *self.charge_equivalence_constraints)],
            format="csr"
        )

        surface_constraints = SparseConstraintMatrix.from_molecules(self.molecules)
        a = scipy.sparse.bmat(
            [[surface_constraints.a, charge_constraint_col],
             [charge_constraint_col.transpose(), None]]
        )
        b = np.r_[surface_constraints.b.toarray(),
                  [c.charge for c in self.charge_sum_constraints],
                  np.zeros(len(self.charge_equivalence_constraints))]
        return SparseConstraintMatrix(a, b)

    def add_constraints_from_charges(self, charges):
        increments = self.get_molecule_increments()
        unconstrained_atoms = [atom for atom in eqv.atoms
                               for eqv in self.charge_equivalence_constraints]
        unconstrained_atoms += self.unconstrained_atoms
        unconstrained_indices = [a.atom_index + increments[a.molecule]
                                 for a in unconstrained_atoms]

        indices = np.arange(self.n_atoms)
        to_constrain = ~np.in1d(indices, unconstrained_indices)
        charges = charges[to_constrain]
        indices = indices[to_constrain]

        for i, q in zip(indices, charges):
            self.add_charge_sum_constraint_from_indices(charge=q, indices=[i])

        self.clean_charge_sum_constraints()
        self.clean_charge_equivalence_constraints()

    def _atoms_from_indices(self, indices):
        return [self._atom_from_index(i) for i in indices]

    def add_charge_sum_constraint_from_indices(self, charge, indices=[]):
        constraint = ChargeSumConstraint(charge=charge,
                                         atoms=self._atoms_from_indices(indices))
        self.charge_sum_constraints.append(constraint)

    def _atom_from_index(self, index):
        i = np.searchsorted(self._n_molecule_atoms, index, side="right") - 1
        return Atom(molecule=self.molecules[i],
                    atom_index=index - self._n_molecule_atoms[i])

    def _index_array(self, array):
        return [array[i:j] for i, j in self._edges]

    def add_sp3_equivalences(self, accepted_n_hs=(2, 3)):
        """
        """
        for mol in self.molecules:
            self._add_mol_sp3_equivalence(mol)

    def _add_mol_sp3_equivalence(self, molecule):
        ch_groups = psi4utils.get_sp3_ch_indices(molecule.qcmol)
        for c, hs in ch_groups.items():
            if len(hs) in accepted_n_hs:
                atoms = [Atom(molecule=molecule, atom_index=i) for i in hs]
                self.charge_equivalence_constraints.append(
                    ChargeEquivalenceConstraint(atoms=atoms)
                )
                self.unconstrained_atoms.append(
                    Atom(molecule=molecule, atom_index=c)
                )
