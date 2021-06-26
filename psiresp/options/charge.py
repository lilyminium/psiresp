from typing import Optional, Union

from .base import options, OptionsBase

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
                 molecule_id: Union[int, "AtomId", Tuple[int, int]] = 1,
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