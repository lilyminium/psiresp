import io

import numpy as np

from .. import utils

class Psi4MolContainerMixin:
    """Mixin class for containing a Psi4 molecule with the .psi4mol attribute"""

    BOHR_TO_ANGSTROM = 0.52917721092

    def __init__(self, psi4mol, *args, name=None, **kwargs):
        if name is not None:
            psi4mol.set_name(name)
        else:
            name = psi4mol.name()
        self.psi4mol = psi4mol
        super(Psi4MolContainerMixin, self).__init__(name=name, **kwargs)


    @property
    def n_atoms(self):
        return self.psi4mol.natom()

    @property
    def indices(self):
        return np.arange(self.n_atoms)

    @property
    def symbols(self):
        return np.array([self.psi4mol.symbol(i) for i in self.indices], dtype=object)
    
    @property
    def coordinates(self):
        return self.psi4mol.geometry().np.astype("float") * self.BOHR_TO_ANGSTROM

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