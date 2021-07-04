from typing import Optional, Union, Tuple, Dict, List
from collections import UserList
import functools
from collections import defaultdict

import numpy as np
import scipy
from pydantic import PrivateAttr

from .. import base

AtomIdType = Optional[Union[int, "AtomId", Tuple[int, int]]]
ChargeConstraintType = Union[Dict[float, List[AtomIdType]],
                             Tuple[float, List[AtomIdType],
                             "ChargeConstraint"]]
ChargeEquivalenceType = Union[List[AtomIdType], "ChargeEquivalence"]


@functools.total_ordering
class AtomId:
    """Atom class to contain the atom number, index, and molecule number

    Parameters
    ----------
    molecule_id: int, AtomId, or tuple of ints
        If this is an integer and `atom_id` is also an integer,
        this is interpreted as the molecule number.
        If this is an integer and `atom_id` not given,
        this is interpreted as the atom number and molecule_id is 1.
        If this is an AtomId, the molecule_id, atom_id, and atom_increment
        are taken from the AtomId and later arguments are ignored.
        If this is a tuple of two integers, the first item is intepreted
        as the molecule_id and the second as the atom_id.
    atom_id: int
        Atom number. Indexed from 1
    atom_increment: int
        Increment to add to the atom ID for the absolute atom ID. This is
        necessary for jobs with multiple molecules, where the third atom
        of the third molecule may be the 12th atom overall if the molecules
        are concatenated.

    Attributes
    ----------
    molecule_id: int
        Molecule number. Indexed from 1
    atom_id: int
        Atom number. Indexed from 1
    atom_increment: int
        Increment to add to the atom ID for the absolute atom ID. This is
        necessary for jobs with multiple molecules, where the third atom
        of the third molecule may be the 12th atom overall if the molecules
        are concatenated.
    absolute_atom_id: int
        Absolute atom ID in the overall job
    absolute_atom_index: int
        Absolute atom index in the overall job

    Examples
    --------
    ::

        >>> AtomId()
        AtomId(atom_id=1, molecule_id=1, atom_increment=0)
        >>> AtomId(2)
        AtomId(atom_id=2, molecule_id=1, atom_increment=0)
        >>> AtomId(2, 3)
        AtomId(atom_id=3, molecule_id=2, atom_increment=0)
        >>> AtomId((2, 3))
        AtomId(atom_id=3, molecule_id=2, atom_increment=0)
        >>> AtomId(AtomId(2, 3, atom_increment=4), atom_increment=6)
        AtomId(atom_id=3, molecule_id=2, atom_increment=4)

    """

    def __init__(self,
                 molecule_id: AtomIdType = None,
                 atom_id: Optional[int] = None,
                 atom_increment: int = 0):
        if isinstance(molecule_id, AtomId):
            atom_id = molecule_id.atom_id
            atom_increment = molecule_id.atom_increment
            molecule_id = molecule_id.molecule_id
        else:
            if atom_id is None:
                if isinstance(molecule_id, (list, tuple)) and len(molecule_id) == 2:
                    atom_id = molecule_id[1]
                    molecule_id = molecule_id[0]
                else:
                    atom_id = molecule_id
                    molecule_id = 1
        self.atom_id = atom_id
        self.molecule_id = molecule_id
        self.atom_increment = atom_increment

    def __repr__(self):
        return (f"AtomId(atom_id={self.atom_id}, "
                f"molecule_id={self.molecule_id}, "
                f"atom_increment={self.atom_increment})")

    def __lt__(self, other):
        if isinstance(other, AtomId):
            other = other.absolute_atom_id
        return self.absolute_atom_id < other

    def __eq__(self, other):
        if isinstance(other, AtomId):
            other = other.absolute_atom_id
        return self.absolute_atom_id == other

    def __hash__(self):
        return hash((self.atom_id, self.molecule_id))

    @property
    def absolute_atom_id(self):
        return self.atom_increment + self.atom_id

    @property
    def absolute_atom_index(self):
        return self.absolute_atom_id - 1

    def copy_with_molecule_id(self,
                              molecule_id: int = 1,
                              atom_increment: int = 0,
                              ) -> "AtomId":
        """Create a copy with a new `molecule_id` and `atom_increment`"""
        new = type(self)(molecule_id=molecule_id, atom_id=self.atom_id)
        new.atom_increment = atom_increment
        return new


class BaseChargeConstraint(UserList):
    """Base class for charge constraints"""

    def __init__(self, atom_ids: List[AtomIdType] = []):
        atom_ids = [AtomId(x) for x in atom_ids]
        atom_ids = sorted(set(atom_ids))
        super().__init__(atom_ids)

    @property
    def atom_ids(self):
        return self.data

    @property
    def absolute_atom_ids(self):
        return np.array([x.absolute_atom_id for x in self.data], dtype=int)

    @property
    def indices(self):
        return self.absolute_atom_ids - 1

    def __len__(self):
        return len(self.atom_ids)

    def copy_atom_ids_to_molecule(self,
                                  molecule_id: int = 1,
                                  atom_increment: int = 0,
                                  ) -> List[AtomId]:
        atom_ids = [aid.copy_with_molecule_id(molecule_id=molecule_id,
                                              atom_increment=atom_increment)
                    for aid in self.atom_ids]
        return atom_ids

    @property
    def molecule_ids(self):
        return [atom.molecule_id for atom in self.atom_ids]

    @molecule_ids.setter
    def molecule_ids(self, value: int):
        for atom in self.atom_ids:
            atom.molecule_id = value

    @property
    def atom_increments(self):
        return [atom.atom_increment for atom in self.atom_ids]

    @atom_increments.setter
    def atom_increments(self, value: int):
        for atom in self.atom_ids:
            atom.atom_increment = value

    def all_molecule_ids_defined(self):
        return all(molid is not None for molid in self.molecule_ids)

    def any_molecule_ids_defined(self):
        return any(molid is not None for molid in self.molecule_ids)

    def some_molecule_ids_defined(self):
        return (self.any_molecule_ids_defined()
                and not self.all_molecule_ids_defined())


class ChargeConstraint(BaseChargeConstraint):
    """Constrain a group of atoms to a specified charge.

    If this constraint is applied, then the sum of the partial atomic
    charges of the specified atoms must sum to the given charge.

    Parameters
    ----------
    charge: float
        Specified charge
    atom_ids: list of AtomId, integers, or tuples of integers
        Specified atoms
    """

    def __init__(self, charge: float = 0, atom_ids: List[AtomIdType] = []):
        self.charge = charge
        super().__init__(atom_ids=atom_ids)

    def __repr__(self):
        return (f"<ChargeConstraint charge={self.charge}, "
                f"indices={self.indices}>")

    @classmethod
    def from_obj(cls, obj: ChargeConstraintType) -> "ChargeConstraint":
        if isinstance(obj, dict):
            if len(obj) != 1:
                raise ValueError("dict must have only one key-value pair "
                                 "in charge: [atom_ids] format.")
            obj = list(obj.items())[0]
        elif isinstance(obj, ChargeConstraint):
            obj = [obj.charge, obj.atom_ids]
        return cls(charge=obj[0], atom_ids=obj[1])

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __hash__(self):
        args = (np.round(self.charge, decimals=3),
                tuple(sorted(set(self.atom_ids))))
        return hash(args)

    def copy_with_molecule_id(self,
                              molecule_id: int = 1,
                              atom_increment: int = 0,
                              ) -> "ChargeConstraint":
        ids = self.copy_atom_ids_to_molecule(molecule_id=molecule_id,
                                             atom_increment=atom_increment)
        return type(self)(charge=self.charge, atom_ids=ids)

    def to_coo_rows(self, n_dim):
        n_items = len(self)
        data = np.ones(n_items)
        row = np.zeros(n_items, dtype=int)
        shape = (1, n_dim)
        return scipy.sparse.coo_matrix((data, (row, self.indices)), shape=shape)

    def to_coo_cols(self, n_dim):
        return self.to_coo_rows(n_dim).transpose()


class ChargeEquivalence(BaseChargeConstraint):
    """Constrain a group of atoms to each have equivalent charge.

    This must contain at least 2 atoms or it doesn't make sense.

    Parameters
    ----------
    atom_ids: list of AtomId, integers, or tuples of integers
        Specified atoms

    """

    def __repr__(self):
        return f"<ChargeEquivalence indices={self.indices}>"

    def __init__(self, atom_ids: List[AtomIdType] = []):
        super().__init__(atom_ids=atom_ids)
        if not len(self.atom_ids) >= 2:
            raise ValueError("Must have at least 2 different atoms in a "
                             "charge equivalence constraint")

    def __add__(self, other):
        return type(self)(np.concatenate([self.atom_ids, other.atom_ids]))

    def __radd__(self, other):
        if other == 0:
            return self
        return other.__add__(self)

    def __eq__(self, other):
        if isinstance(other, type(self)):
            other = other.indices
        return set(list(self.indices)) == set(list(other))

    def __hash__(self):
        return hash(tuple(sorted(set(self.atom_ids))))

    def copy_with_molecule_id(self,
                              molecule_id: int = 1,
                              atom_increment: int = 0,
                              ) -> "ChargeEquivalence":
        ids = self.copy_atom_ids_to_molecule(molecule_id=molecule_id,
                                             atom_increment=atom_increment)
        return type(self)(atom_ids=ids)

    def to_coo_rows(self, n_dim):
        n_items = len(self) - 1
        data = np.concatenate([-np.ones(n_items), np.ones(n_items)])
        row = np.tile(np.arange(n_items, dtype=int), 2)
        col = np.concatenate([self.indices[:-1], self.indices[1:]])
        shape = (n_items, n_dim)
        return scipy.sparse.coo_matrix((data, (row, col)), shape=shape)

    def to_coo_cols(self, n_dim):
        return self.to_coo_rows(n_dim).transpose()


class ChargeConstraintOptions(base.Model):
    """Options for setting charge constraints and charge equivalence constraints.

    Parameters
    ----------
    charge_constraints: list of dicts, tuples, or ChargeConstraints
        This is a list of all inputs accepted by the
        :class:`ChargeConstraint` class.
        It will be used to create constraints for a group of atoms to
        the given charge.
    charge_equivalences: list of lists, tuples, or ChargeEquivalences
        This is a list of all inputs accepted by the
        :class:`ChargeEquivalence` class.
        It will be used to create constraints so that each atom in the
        given group is constrained to the same charge.

    """
    charge_constraints: List[ChargeConstraint] = []
    charge_equivalences: List[ChargeEquivalence] = []
    symmetric_methyls: bool = True
    symmetric_methylenes: bool = True
    _do_not_constrain_ids: List[int] = PrivateAttr(default_factory=list)

    # def __post_init__(self):
    #     self.clean_charge_constraints()
    #     self.clean_charge_equivalences()

    @property
    def n_charge_constraints(self):
        return len(self.charge_constraints)

    @property
    def n_charge_equivalences(self):
        return len(self.charge_equivalences)

    def iterate_over_constraints(self):
        "Iterate first over charge constraints and then charge equivalences"
        for item in self.charge_constraints:
            yield item
        for item in self.charge_equivalences:
            yield item

    def _unite_overlapping_equivalences(self):
        """Join ChargeEquivalence constraints with overlapping atoms"""
        equivalences = defaultdict(set)
        for chrequiv in self.charge_equivalences:
            atom_set = set(chrequiv.atom_ids)
            for atom in atom_set:
                equivalences[atom] |= atom_set

        chrequivs = {tuple(sorted(x)) for x in equivalences.values()}
        self.charge_equivalences = [ChargeEquivalence(x) for x in chrequivs]

    def _get_single_atom_charge_constraints(self) -> Dict[AtomId, float]:
        """Get ChargeConstraints with only one atom as a dict"""
        single_charges = {}
        for constr in self.charge_constraints:
            if len(constr.atom_ids) == 1:
                atom_id = constr.atom_ids[0]
                if atom_id in single_charges:
                    err = ("Found conflicting charge constraints for "
                           f"atom {atom_id}, constrained to both "
                           f"{single_charges[atom_id]} and {constr.charge}")
                    raise ValueError(err)
                single_charges[atom_id] = constr.charge
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
        for i_eq, chrequiv in enumerate(self.charge_equivalences):

            try:
                indices, charges = zip(*[(i, single_charges[x])
                                         for i, x in enumerate(chrequiv.atom_ids)
                                         if x in single_charges])
            except ValueError:
                continue

            if len(charges) > 1:
                for i in indices[::-1]:
                    # TODO: silently delete or raise an error?
                    del chrequiv[i]
            # every atom in the equivalence is constrained to the same charge
            # this is redundant and should get removed
            # or can result in singular matrices
            elif len(charges) == len(chrequiv.atom_ids):
                redundant.append(i_eq)
        for i in redundant[::-1]:
            del self.equivalences[i]

    def clean_charge_equivalences(self):
        """Clean the ChargeEquivalence constraints.

        1. Join charge equivalence constraints with overlapping atoms
        2. Remove atoms from charge equivalences if they are constrained
        to different charges, and remove charge equivalence constraints
        if all atoms are constrained to the same charge (so it is redundant)
        """
        self._unite_overlapping_equivalences()
        self._remove_incompatible_and_redundant_equivalent_atoms()
        self.charge_equivalences = sorted(self.charge_equivalences)

    def clean_charge_constraints(self):
        """Clean the ChargeConstraints.

        1. Removes duplicates
        2. Checks that there are no atoms constrained to two different charges
        """
        # remove duplicates
        self.charge_constraints = list(set(self.charge_constraints))
        # this will check for duplicate conflicting charges as a side effect
        self._get_single_atom_charge_constraints()
        self.charge_constraints = sorted(self.charge_constraints)

    def get_constraint_matrix(self, a_matrix, b_matrix):
        """Create full constraint matrix from input matrices and charge constraints.

        A and B are the matrices used to solve Ax = B.

        Parameters
        ----------
        a_matrix: numpy.ndarray
            Matrix of shape (N, N)
        b_matrix: numpy.ndarray
            Matrix of shape (N,)

        Returns
        -------
        A: numpy.ndarray
            Overall matrix of constraints, shape (M, M).
            M = N + number_of_charge_constraints + number_of_equivalent_atom_pairs
        B: numpy.ndarray
            Overall solution vector, shape (M,)

        """
        n_dim = a_matrix.shape[0]
        col_constraints = [c.to_coo_cols(n_dim)
                           for c in self.iterate_over_constraints()]
        col_block = scipy.sparse.hstack(col_constraints, format="coo")
        if col_block.shape[0] == 0:
            return a_matrix, b_matrix
        a_block = scipy.sparse.coo_matrix(a_matrix)
        a_sparse = scipy.sparse.bmat([[a_block, col_block],
                                      [col_block.transpose(), None]])
        # b_dense = np.r_[b_matrix, [c.charge for c in self.charge_constraints]]
        b_dense = b_matrix
        b_sparse = np.zeros(a_sparse.shape[0])
        b_sparse[:len(b_dense)] = b_dense
        # print(col_block.toarray().T, b_sparse[-6:])
        return a_sparse.toarray(), b_sparse

    def add_sp3_equivalences(self, sp3_ch_ids: Dict[int, List[int]] = {}):
        """
        Add ChargeEquivalences for the hydrogens attached to sp3 carbons

        This will add methyls if ``symmetric_methyls`` is True,
        and methylenes if ``symmetric_methylenes`` is True.

        Parameters
        ----------
        sp3_ch_ids: dictionary of {int: list[int]}
            A dictionary of atom numbers.
            Atom numbers are indices, indexed from 1.
            Keys are the atom numbers of carbons with 4 bonds.
            Values are the numbers of the hydrogens bonded to these carbons.
        """
        accepted = []
        if self.symmetric_methyls:
            accepted.append(3)
        if self.symmetric_methylenes:
            accepted.append(2)
        if not accepted:
            return
        for c, hs in sp3_ch_ids.items():
            if len(hs) in accepted:
                self.charge_equivalences.append(ChargeEquivalence(hs))
                self._do_not_constrain_ids.append(c)

    def add_stage_2_constraints(self, charges=[], ):
        """Add ChargeConstraints restraining atoms to the given charges,
        if they are not in charge equivalence constraints.

        Parameters
        ----------
        charges: iterable of floats
            Charges
        """
        charges = np.asarray(charges)
        unconstrained = [i for eq in self.charge_equivalences
                         for i in eq.atom_ids]
        unconstrained += self._do_not_constrain_ids
        equivalent_atom_ids = np.array(unconstrained)
        all_atom_ids = np.arange(len(charges), dtype=int) + 1
        mask = ~np.in1d(all_atom_ids, equivalent_atom_ids)

        for q, a in zip(charges[mask], all_atom_ids[mask]):
            constr = ChargeConstraint(charge=q, atom_ids=[a])
            self.charge_constraints.append(constr)

        self.clean_charge_constraints()
        self.clean_charge_equivalences()
