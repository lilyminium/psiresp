import abc
import os
from dataclasses import dataclass, field, Field
from collections import UserDict
import pathlib
import tempfile
import contextlib

import psi4

from .utils import (split_docstring_around_parameters,
                    get_parameter_docstrings,
                    extend_new_class_parameters,
                    add_option_to_init,
                    create_option_property)

from ..options import IOOptions


class ContainsOptionsMeta(type):

    def __init__(cls, name, bases, clsdict):
        docstring = cls.__doc__
        init = cls.__init__
        before, params, after = split_docstring_around_parameters(docstring)
        for name, value in clsdict.items():
            if isinstance(value, Field) and issubclass(value.default_factory, AttrDict):
                base = value.default_factory
                if not hasattr(base, "__dataclass_fields__"):
                    continue
                base_name = base.__name__
                if not base_name.endswith("Options"):
                    continue
                docstrings = get_parameter_docstrings(base)
                extend_new_class_parameters(base, params)
                base_argname = base_name[:-7].lower() + "_" + "options"
                init = add_option_to_init(base_argname, base)(init)
                for field in base.__dataclass_fields__:
                    prop = create_option_property(field, base_argname, docstrings)
                    setattr(cls, field, prop)
        cls.__doc__ = "\n".join([before, "", "\n".join(params), after])
        cls.__init__ = init


# @dataclass
# class IOBase(metaclass=ContainsOptionsMeta):
#     name: str = "mol"
#     io_options: IOOptions = IOOptions()

    
class ContainsOptionsBase(metaclass=ContainsOptionsMeta):
    pass


BOHR_TO_ANGSTROM = 0.52917721092

@dataclass
class MoleculeBase(ContainsOptionsBase):

    psi4mol: psi4.core.Molecule
    io_options: IOOptions = field(default_factory=IOOptions)

    def __post_init__(self):
        if self.name:
            self.psi4mol.set_name(self.name)
        else:
            self.name = self.psi4mol.name()
    
    @property
    def n_atoms(self):
        return self.psi4mol.natom()

    @property
    def indices(self):
        return np.arange(self.n_atoms)

    @property
    def symbols(self):
        return np.array([self.psi4mol.symbol(i) for i in self.indices],
                        dtype=object)
    
    @property
    def coordinates(self):
        bohr = self.psi4mol.geometry().np.astype("float")
        return bohr * BOHR_TO_ANGSTROM
    
    def to_mda(self):
        """Create a MDAnalysis.Universe from molecule
        
        Returns
        -------
        MDAnalysis.Universe
        """
        import MDAnalysis as mda
        mol = utils.psi42xyz(self.psi4mol)
        u = mda.Universe(io.StringIO(mol), format="XYZ")
        return u
    
    def write(self, filename):
        """Write molecule to file.

        This uses the MDAnalysis engine to write out to different
        formats.
        
        Parameters
        ----------
        filename: str
            Filename to write the molecule to.
        """
        u = self.to_mda()
        u.atoms.write(filename)