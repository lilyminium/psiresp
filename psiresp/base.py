import functools
import os
import pathlib
import contextlib
import tempfile

import numpy as np

from .options import IOOptions


def datafile(func=None, filename=None):
    """Try to load data from file. If not found, saves data to same path"""

    if func is None:
        return functools.partial(datafile, filename=filename)

    fname = func.__name__
    if fname.startswith("compute_"):
        fname = fname.split("compute_", maxsplit=1)[1]

    filename = filename if filename else fname + ".dat"

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        fn = self.name + "_" + filename
        data, path = self.io_options.try_load_data(fn)
        if data is not None:
            return data
        data = func(self, *args, **kwargs)
        comments = None
        if path.endswith("npy"):
            try:
                data, comments = data
            except ValueError:
                pass
        self.io_options.save_data(data, path, comments=comments)
        return data

    return wrapper


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


class IOBase:
    """Base class for containing IOOptions
    
    Parameters
    ----------
    name: str (optional)
        The name of this instance
    io_options: psiresp.IOOptions
        input/output options

    Attributes
    ----------
    name: str (optional)
        The name of this instance
    io_options: psiresp.IOOptions
        input/output options
    """
    def __init__(self, name=None, io_options=IOOptions()):
        self.name = name
        self.io_options = io_options


    @property
    def io_options(self):
        return self._io_options

    @contextlib.contextmanager
    def get_subfolder(self):
        cwd = pathlib.Path.cwd()
        if self.io_options.write_to_files:
            path = cwd / self.name
            path.mkdir(exist_ok=True)
        else:
            path = tempfile.TemporaryDirectory()
        
        # os.chdir(path)
        try:
            yield path.name
        finally:
            # os.chdir(cwd)
            if not self.io_options.write_to_files:
                path.cleanup()

    @io_options.setter
    def io_options(self, options):
        self._io_options = IOOptions(**options)
