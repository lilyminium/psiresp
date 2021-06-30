
import io
import functools
import pathlib
import tempfile
import contextlib
import os
from dataclasses import dataclass, field, Field

import psi4
import numpy as np

from .options import ContainsOptionsBase, IOOptions, QMOptions

from . import constants, psi4utils

def datafile(func=None, path=None):
    """Try to load data from file. If not found, saves data to same path"""

    if func is None:
        return functools.partial(datafile, path=filename)
    
    if path is None:
        fname = func.__name__
        if fname.startswith("compute_"):
            fname = fname.split("compute_", maxsplit=1)[1]
        path = fname + ".dat"

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        path = path.format(name=self.name, path=self.path)
        return self.io_options.try_datafile(path, wrapper, self, *args, **kwargs)
    return wrapper

@dataclass
class MoleculeBase(ContainsOptionsBase):

    psi4mol: psi4.core.Molecule
    name: str = "mol"
    io_options: IOOptions = field(default_factory=IOOptions)

    def __post_init__(self):
        if self.name:
            self.psi4mol.set_name(self.name)
        else:
            self.name = self.psi4mol.name()
    
    @property
    def path(self):
        return pathlib.Path(self.name)

    @contextlib.contextmanager
    def directory(self):
        cwd = pathlib.Path.cwd()
        if self.io_options.save_output:
            path = self.path
            path.mkdir(parents=True, exist_ok=True)
        else:
            path = tempfile.TemporaryDirectory()
        
        try:
            os.chdir(path)
            yield path.name
        finally:
            os.chdir(cwd)
            if not self.save_output:
                path.cleanup()
    
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
    def charge(self):
        return self.psi4mol.molecular_charge()

    @charge.setter
    def charge(self, value):
        if value != self.psi4mol.molecular_charge():
            self.psi4mol.set_molecular_charge(value)
            self.psi4mol.update_geometry()
    
    @property
    def multiplicity(self):
        return self.psi4mol.multiplicity()
    
    @multiplicity.setter
    def multiplicity(self, value):
        # can cause issues if we set willy-nilly
        if value != self.psi4mol.multiplicity():
            self.psi4mol.set_multiplicity(value)
            self.psi4mol.update_geometry()

    
    @property
    def coordinates_in_bohr(self):
        return self.psi4mol.geometry().np.astype("float")

    @property
    def coordinates(self):
        return self.coordinates_in_bohr * constants.BOHR_TO_ANGSTROM
    
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

    def clone(self, name: str = None):
        mol = self.psi4mol.clone()
        if name is not None:
            mol.set_name(name)
        else:
            name = f"{self.name}_copy"
        
        state = self.to_dict()
        state.pop("psi4mol")
        state["name"] = name

        return type(self)(mol, **state)

    