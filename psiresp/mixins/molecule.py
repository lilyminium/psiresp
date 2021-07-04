import io
from typing import Optional, Any
import pathlib

import numpy as np
from pydantic import BaseModel
import psi4

from .. import base, psi4utils, utils
from .io import IOMixin


class MoleculeMixin(IOMixin):
    """Class that contains a Psi4 molecule and a name

    Parameters
    ----------
    psi4mol: psi4.core.Molecule
        Psi4 molecule
    name: str
        Name

    Attributes
    ----------
    psi4mol: psi4.core.Molecule
        Psi4 molecule
    name: str
        Name
    path: pathlib.Path
        Associated directory
    n_atoms: int
        Number of atoms in molecule
    indices: numpy.ndarray of integers
        Indices for molecule atoms
    symbols: list of str
        Atomic element symbols for molecule
    charge: float
        Charge of molecule
    multiplicity: int
        Multiplicity of molecule
    psi4mol_geometry_in_bohr: bool
        Whether the Psi4 molecule geometry is in units of Bohr
    coordinates: numpy.ndarray of floats
        Psi4 molecule coordinates in angstrom

    """

    psi4mol: psi4.core.Molecule
    name: Optional[str] = None

    # def __post_init__(self):
    #     if self.name:
    #         self.psi4mol.set_name(self.name)
    #     else:
    #         self.name = self.psi4mol.name()

    def __init__(self, *args, **kwargs):
        if args and len(args) == 1 and "psi4mol" not in kwargs:
            kwargs["psi4mol"] = args[0]
            args = tuple()
        super().__init__(*args, **kwargs)
        self.psi4mol.set_name(self.name)

    # @property
    # def name(self):
    #     return self.psi4mol.name()

    # @name.setter
    # def name(self, value):
    #     if value is None:
    #         value = self.psi4mol.name()
    #     self.psi4mol.set_name(value)

    @property
    def path(self):
        if self.directory_path is None:
            return self.default_path
        return self.directory_path

    @property
    def default_path(self):
        return pathlib.Path(self.name)

    @property
    def n_atoms(self):
        return self.psi4mol.natom()

    @property
    def indices(self):
        return np.arange(self.n_atoms, dtype=int)

    @property
    def symbols(self):
        return [self.psi4mol.symbol(i) for i in self.indices]

    # @property
    # def charge(self):
    #     return self.psi4mol.molecular_charge()

    # @charge.setter
    # def charge(self, value):
    #     if value != self.psi4mol.molecular_charge():
    #         self.psi4mol.set_molecular_charge(value)
    #         self.psi4mol.update_geometry()

    # @property
    # def multiplicity(self):
    #     return self.psi4mol.multiplicity()

    # @multiplicity.setter
    # def multiplicity(self, value):
    #     # can cause issues if we set willy-nilly
    #     if value != self.psi4mol.multiplicity():
    #         self.psi4mol.set_multiplicity(value)
    #         self.psi4mol.update_geometry()

    @property
    def psi4mol_geometry_in_bohr(self):
        return self.psi4mol.units() == "Bohr"

    @property
    def coordinates(self):
        geometry = self.psi4mol.geometry().np.astype("float")
        # if self.psi4mol_geometry_in_bohr:
        #     geometry *= constants.BOHR_TO_ANGSTROM
        return geometry * utils.BOHR_TO_ANGSTROM

    def to_mda(self):
        """Create a MDAnalysis.Universe from molecule

        Returns
        -------
        MDAnalysis.Universe
        """
        import MDAnalysis as mda
        mol = psi4utils.psi4mol_to_xyz_string(self.psi4mol)
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

    def clone(self, name: str = None) -> "MoleculeMixin":
        """Copy to new instance"""
        mol = self.psi4mol.clone()
        if name is not None:
            mol.set_name(name)
        else:
            name = f"{self.name}_copy"

        state = self.to_kwargs(psi4mol=mol, name=name)
        return type(self)(**state)
