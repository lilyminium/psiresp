from typing import Optional, Union, Tuple, Dict, List
from collections import UserList

import numpy as np
import scipy

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
        return scipy.sparse.coo_matrix(data, (row, self.indices), shape=shape)

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
        data = np.concatenate([np.ones(n_items), -np.ones(n_items)])
        row = np.tile(np.arange(n_items, dtype=int), 2)
        col = np.concatenate([self.indices[:-1], self.indices[1:]])
        shape = (n_items, n_dim)
        return scipy.sparse.coo_matrix(data, (row, col), shape=shape)

    def to_coo_cols(self, n_dim):
        return self.to_coo_rows(n_dim).transpose()
