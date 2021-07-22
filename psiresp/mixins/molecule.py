from typing import Optional
import pathlib

import numpy as np
import psi4
from pydantic import Field, validator

from .. import utils
from ..utils import psi4utils
from .io import IOMixin


class MoleculeMixin(IOMixin):
    """Class that contains a Psi4 molecule and a name

    Attributes
    ----------
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

    psi4mol: psi4.core.Molecule = Field(description="Psi4 molecule")
    name: Optional[str] = Field(
        default=None,
        description=("Name of this instance. "
                     "This is mostly used to find or create directories for "
                     "saving QM and intermediate files."),
    )

    @classmethod
    def from_molfile(cls, molfile: str, **kwargs):
        """Create class from molecule file

        Parameters
        ----------
        molfile: str
            filename containing the molecule specification. This will
            get automatically parsed if it is a valid PDB, XYZ, MOL, or MOL2
            file, or has a suffix that can get parsed by MDAnalysis. This
            must only contain *one* molecule specification; multiple
            molecules (e.g. in the PDB format) are not supported.
        **kwargs:
            Further arguments for initialization of the class
            (see class docstring)
        """
        mols = psi4utils.psi4mols_from_file(molfile)
        if len(mols) != 1:
            raise ValueError("Must provide only one molecule specification. "
                             f"Given: {len(mols)}")
        return cls(psi4mol=mols[0], **kwargs)

    @validator("psi4mol")
    def validate_psi4mol(cls, v):
        if isinstance(v, str):
            v = psi4.core.Molecule.from_string(v, dtype="psi4")
        assert isinstance(v, psi4.core.Molecule)
        return v

    def __init__(self, *args, **kwargs):
        if args and len(args) == 1 and "psi4mol" not in kwargs:
            kwargs["psi4mol"] = args[0]
            args = tuple()
        super().__init__(*args, **kwargs)
        if self.name is not None:
            self.psi4mol.set_name(self.name)
        else:
            self.name = self.psi4mol.name()

    def __getstate__(self):
        psi4mol = self.psi4mol.to_string(dtype="psi4")
        state = super().__getstate__()
        state_dict = dict(state["__dict__"])
        state_dict["psi4mol"] = psi4mol
        state["__dict__"] = state_dict
        return state

    def __setstate__(self, state):
        state_dict = state["__dict__"]
        state = {k: v for k, v in state.items() if k != "__dict__"}
        try:
            molstring = state_dict.pop("psi4mol")
        except KeyError:
            raise TypeError("State must have a `psi4mol` defined")
        psi4mol = psi4.core.Molecule.from_string(molstring, dtype="psi4")
        state_dict["psi4mol"] = psi4mol
        state["__dict__"] = state_dict
        super().__setstate__(state)

    # TODO: confuzzle with pydantic's aliasing
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
        from io import StringIO
        import MDAnalysis as mda
        mol = psi4utils.psi4mol_to_xyz_string(self.psi4mol)
        u = mda.Universe(StringIO(mol), format="XYZ")
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
