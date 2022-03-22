import functools
from typing import List, Dict, Set, Tuple
from collections import defaultdict
import warnings

import numpy as np
from pydantic import Field

from . import base
from .molecule import Atom, Molecule


@functools.total_ordering
class BaseChargeConstraint(base.Model):
    """Base class for charge constraints"""

    atoms: Set[Atom] = Field(
        default_factory=set,
        description="Atoms involved in the constraint"
    )

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

    def get_atom_indices(self, molecule_increments: Dict[int, List[int]] = {}):
        indices = [atom.index + molecule_increments.get(hash(atom.molecule), [0])[0]
                   for atom in self.atoms]
        return np.array(sorted(indices), dtype=int)


class ChargeSumConstraint(BaseChargeConstraint):
    """Constrain a group of atoms to a specified charge.

    If this constraint is applied, then the sum of the partial atomic
    charges of the specified atoms must sum to the given charge.
    """

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
                          molecule_increments: Dict[int, List[int]] = {}):
        row = np.zeros((1, n_dim))
        indices = self.get_atom_indices(molecule_increments)
        row[0, indices] = 1
        return row

    def __hash__(self):
        return hash((frozenset(self.atoms), self.charge))


class ChargeEquivalenceConstraint(BaseChargeConstraint):
    """Constrain a group of atoms to each have equivalent charge.

    This must contain at least 2 atoms or it doesn't make sense.
    """

    def to_row_constraint(self, n_dim: int,
                          molecule_increments: Dict[int, List[int]] = {}):
        indices = self.get_atom_indices(molecule_increments)
        return self._convert_indices_to_constraint_rows(n_dim, indices)

    @property
    def charge(self):
        return None

    @staticmethod
    def _convert_indices_to_constraint_rows(n_dim, indices):
        n_items = len(indices) - 1
        rows = np.zeros((n_items, n_dim))
        for i, (j, k) in enumerate(zip(indices[:-1], indices[1:])):
            rows[i][j] = -1
            rows[i][k] = 1
        return rows


class BaseChargeConstraintOptions(base.Model):
    charge_sum_constraints: List[ChargeSumConstraint] = []
    charge_equivalence_constraints: List[ChargeEquivalenceConstraint] = []
    split_conformers: bool = Field(
        default=False,
        description="Treat conformers separately, instead of combining the restraint matrices"
    )
    constrain_methyl_hydrogens_between_conformers: bool = Field(
        default=False,
        description=(
            "Whether to constrain methyl/ene hydrogens as equivalent between conformers. "
            "This has no effect if `split_conformers=False`."
        )
    )

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

        unique_sets = []
        for group in equivalences.values():
            for existing in unique_sets:
                if group & existing:
                    existing |= group
                    break
            else:
                unique_sets.append(group)

        self.charge_equivalence_constraints = [ChargeEquivalenceConstraint(atoms=sorted(x))
                                               for x in unique_sets]

    def _get_single_atom_charge_constraints(self) -> Dict[Atom, float]:
        """Get ChargeConstraints with only one atom as a dict"""
        single_charges = {}
        for constr in self.charge_sum_constraints:
            if len(constr.atoms) == 1:
                atom = list(constr.atoms)[0]
                if atom in single_charges and not np.allclose(single_charges[atom], constr.charge, atol=1e-4):
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
        charge_ids = [hash(x) for x in single_charges]
        redundant = []
        for i, constraint in enumerate(self.charge_sum_constraints):
            if len(constraint.atoms) > 1:
                if all(hash(atom) in charge_ids for atom in constraint.atoms):
                    redundant.append(i)
        for i in redundant[::-1]:
            self.charge_sum_constraints.pop(i)

    def clean_charge_equivalence_constraints(self):
        """Clean the ChargeEquivalence constraints.

        1. Join charge equivalence constraints with overlapping atoms
        2. Remove atoms from charge equivalences if they are constrained
            to different charges, and remove charge equivalence constraints
            if all atoms are constrained to the same charge (so it is redundant)

        """
        self._unite_overlapping_equivalences()
        self._remove_incompatible_and_redundant_equivalent_atoms()
        constraint_set = set([
            constraint
            for constraint in self.charge_equivalence_constraints
            if len(constraint.atoms) > 1
        ])
        self.charge_equivalence_constraints = sorted(constraint_set)

    def clean_charge_sum_constraints(self):
        self._remove_redundant_charge_constraints()
        self.charge_sum_constraints = sorted(set(self.charge_sum_constraints))


class ChargeConstraintOptions(BaseChargeConstraintOptions):
    """Options for setting charge constraints and charge equivalence constraints."""
    symmetric_methyls: bool = Field(
        default=True,
        description=("Whether to constrain methyl hydrogens around "
                     "an sp3 carbon to equivalent charge")
    )
    symmetric_methylenes: bool = Field(
        default=True,
        description=("Whether to constrain methylene hydrogens around "
                     "a carbon to equivalent charge")
    )
    symmetric_atoms_are_equivalent: bool = Field(
        default=False,
        description=("Whether to constrain atoms to have equivalent charges "
                     "if they are symmetric in the graph representation. "
                     "3D coordinates are *not* taken into account.")
    )

    def __post_init__(self, **kwargs):
        warnings.warn(
            (
                "`symmetric_atoms_are_equivalent` will be set to False "
                "by default for now, as it is a new feature. "
                "It will be set to True by default in the future"
            ),
            FutureWarning)
        return super().__post_init__(**kwargs)


class MoleculeChargeConstraints(BaseChargeConstraintOptions):
    molecules: List[Molecule] = []
    unconstrained_atoms: List[Atom] = []

    _n_atoms: int
    _n_total_atoms: int
    _n_conformers: List[int]
    _n_molecule_atoms: np.ndarray
    _molecule_increments: Dict[int, List[int]]
    _edges: List[Tuple[int, int]]

    def __post_init__(self, **kwargs):
        super().__post_init__(**kwargs)
        self._n_atoms = sum([mol.n_atoms for mol in self.molecules])
        self.clean_charge_sum_constraints()
        self.clean_charge_equivalence_constraints()
        self._generate_molecule_increments()

    def _generate_molecule_increments(self):
        self._molecule_increments = {}
        increment = 0
        for mol in self.molecules:
            n_atoms = mol.n_atoms
            confs = mol.conformers
            if not self.split_conformers:
                confs = confs[:1]
            inc_list = []
            for _ in confs:
                inc_list.append(increment)
                increment += n_atoms
            self._molecule_increments[hash(mol)] = inc_list
        self._n_total_atoms = increment
        self._generate_edges()

    def _generate_edges(self):
        increments = list(self._molecule_increments.values())
        self._n_molecule_atoms = np.array([x[0] for x in increments] + [self._n_total_atoms])
        self._edges = []
        for mol in self.molecules:
            starter = self._molecule_increments[hash(mol)][0]
            ender = starter + mol.n_atoms
            self._edges.append((starter, ender))

    @property
    def _constraint_conformers(self):
        if self.split_conformers:
            return [conf for mol in self.molecules for conf in mol.conformers]
        return self.molecules

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

        dct = {
            k: getattr(charge_constraints, k)
            for k in [
                "split_conformers",
                "constrain_methyl_hydrogens_between_conformers",
            ]
        }

        constraints = cls(
            charge_sum_constraints=sums,
            charge_equivalence_constraints=eqvs,
            molecules=molecules,
            **dct
        )

        accepted = []
        if charge_constraints.symmetric_methyls:
            accepted.append(3)
        if charge_constraints.symmetric_methylenes:
            accepted.append(2)
        if accepted:
            constraints.add_sp3_equivalences(accepted)
        if charge_constraints.symmetric_atoms_are_equivalent:
            constraints.add_symmetry_equivalences()
        return constraints

    def to_a_col_constraints(self) -> List[np.ndarray]:
        n_dim = self._n_total_atoms + len(self._constraint_conformers)

        # include legitimate constraints within a conformer / between conformers
        constraints = [
            con.to_row_constraint(
                n_dim=n_dim,
                molecule_increments=self._molecule_increments,
            ).T
            for con in self.charge_sum_constraints
        ]
        if self.split_conformers:
            for constraint in self.charge_sum_constraints:
                if len(constraint.atoms) == 1:
                    atom = list(constraint.atoms)[0]
                    molhash = hash(atom.molecule)
                    increments = self._molecule_increments[molhash][1:]
                    for inc in increments:
                        col = np.zeros((n_dim, 1))
                        col[atom.index + inc] = 1
                        constraints.append(col)
        for constraint in self.charge_equivalence_constraints:
            constraints.append(
                constraint.to_row_constraint(
                    n_dim=n_dim,
                    molecule_increments=self._molecule_increments,
                ).T)

        # when treating split conformers,
        # single-atom pins shouldn't be equivalenced.
        # makes them weird.
        sum_constraints = defaultdict(set)
        for constr in self.charge_sum_constraints:
            if len(constr.atoms) > 1:
                continue
            for atom in constr.atoms:
                molhash = hash(atom.molecule)
                sum_constraints[molhash].add(atom.index)

        # add inter-conformer constraints
        if self.split_conformers:
            for mol in self.molecules:
                molhash = hash(mol)
                increments = np.array(self._molecule_increments[hash(mol)], dtype=int)
                indices = [
                    i for i in list(range(mol.n_atoms))
                    if i not in sum_constraints[molhash]
                ]
                if not self.constrain_methyl_hydrogens_between_conformers:
                    h_indices = [
                        i for group in mol.get_sp3_ch_indices().values()
                        for i in group
                    ]

                    indices = [
                        i
                        for i in indices
                        if i not in h_indices
                    ]

                for i in indices:
                    col = ChargeEquivalenceConstraint._convert_indices_to_constraint_rows(
                        n_dim, increments + i
                    ).T
                    if col.shape[1]:
                        constraints.append(col)
        return constraints

    def to_b_constraints(self):
        b = [constr.charge for constr in self.charge_sum_constraints]
        if self.split_conformers:
            for constraint in self.charge_sum_constraints:
                if len(constraint.atoms) == 1:
                    atom = list(constraint.atoms)[0]
                    molhash = hash(atom.molecule)
                    increments = self._molecule_increments[molhash][1:]
                    for _ in increments:
                        b.append(constraint.charge)
        return np.array(b)

    def add_constraints_from_charges(self, charges: np.ndarray):
        """Add ChargeSumConstraints restraining atoms to the given charges,
        if they are not in existing charge equivalence constraints,
        and not in ``self.unconstrained_atoms``.

        Parameters
        ----------
        charges: np.ndarray of floats
            Charges of atoms. This should be at least as long as the
            total number of atoms in ``self.molecules``
        """
        unconstrained_atoms = [atom
                               for eqv in self.charge_equivalence_constraints
                               for atom in eqv.atoms]
        unconstrained_atoms += self.unconstrained_atoms

        unconstrained_indices = [a.index + self._molecule_increments[hash(a.molecule)][0]
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

    def get_sp3_equivalences(self, accepted_n_hs: Tuple[int, ...] = (2, 3)):
        equivalent_atoms = {"H": [], "C": []}
        for mol in self.molecules:
            ch_groups = mol.get_sp3_ch_indices()
            for c, hs in ch_groups.items():
                if len(hs) in accepted_n_hs:
                    atoms = [Atom(molecule=mol, index=i) for i in hs]
                    equivalent_atoms["H"].append(atoms)
                    equivalent_atoms["C"].append(Atom(molecule=mol, index=c))
        return equivalent_atoms

    def add_sp3_equivalences(self, accepted_n_hs: Tuple[int, ...] = (2, 3)):
        """
        Add ChargeEquivalenceConstraints for the hydrogens attached to sp3 carbons

        This will add methyls if 2 is in ``accepted_n_hs``, and
        and methylenes if 3 is in ``accepted_n_hs``.

        Parameters
        ----------
        accepted_n_hs:
            Number of Hs around a carbon to symmetrize
        """
        equivalent_atoms = self.get_sp3_equivalences(accepted_n_hs=accepted_n_hs)
        self.charge_equivalence_constraints.extend(
            [
                ChargeEquivalenceConstraint(atoms=atoms)
                for atoms in equivalent_atoms["H"]
            ]
        )
        self.unconstrained_atoms.extend(equivalent_atoms["C"])

    def add_symmetry_equivalences(self):
        for mol in self.molecules:
            for atoms in mol.get_symmetric_atoms():
                self.charge_equivalence_constraints.append(
                    ChargeEquivalenceConstraint(atoms=atoms)
                )

    def prepare_stage_1_constraints(self):
        # heavy atoms, heavy Hs equivalenced
        # basically remove any constraints that are only methyls
        equivalent_atoms = self.get_sp3_equivalences()
        all_atoms = {h for eq in equivalent_atoms["H"] for h in eq}
        all_atoms |= set(equivalent_atoms["C"])

        self.charge_equivalence_constraints = [
            constraint
            for constraint in self.charge_equivalence_constraints
            if not set(constraint.atoms).issubset(all_atoms)
        ]

    def prepare_stage_2_constraints(self):
        # keep only methyl/ene constraints
        return

        # equivalent_atoms = self.get_sp3_equivalences()
        # all_atoms = {h for eq in equivalent_atoms["H"] for h in eq}
        # all_atoms |= set(equivalent_atoms["C"])

        # self.charge_equivalence_constraints = [
        #     constraint
        #     for constraint in self.charge_equivalence_constraints
        #     if set(constraint.atoms).issubset(all_atoms)
        # ]
