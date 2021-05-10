import functools
import os

import numpy as np

from .utils import try_load_data, save_data
from .options import IOOptions


class Psi4MolContainerMixin:

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
        fn = os.path.join(self.directory, self.name + "_" + filename)
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

class IOBase:

    def __init__(self, name=None, io_options=IOOptions()):
        self.name = name
        self.io_options = io_options

    @property
    def io_options(self):
        return self._io_options

    @io_options.setter
    def io_options(self, options):
        self._io_options = IOOptions(**options)