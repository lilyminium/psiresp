import logging
import pathlib
import os
import tempfile
import functools
import abc
import contextlib
from typing import Union, Optional, Callable

import pandas as pd
import numpy as np


from .. import base

logger = logging.getLogger(__name__)

Data = Union[np.ndarray, pd.DataFrame]
Path = Union[pathlib.Path, str]


def datafile(func: Optional[Callable] = None,
             filename: Optional[str] = None,
             ) -> Callable:
    """Try to load data from file. If not found, saves data to same path"""

    if func is None:
        return functools.partial(datafile, filename=filename)

    if filename is None:
        fname = func.__name__
        if fname.startswith("compute_"):
            fname = fname.split("compute_", maxsplit=1)[1]
        filename = fname + ".dat"

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        return self.try_datafile(filename, wrapper, self, *args, **kwargs)
    return wrapper


def load_text(file: str):
    """Load text from file"""
    with open(file, "r") as f:
        return f.read()


def load_data(path: Path) -> Data:
    """
    Parameters
    ----------
    path: pathlib.Path or str
        Data path

    Returns
    -------
    data: numpy.ndarray or pd.DataFrame
        numpy.ndarray or pd.DataFrame
    """
    path = str(path)
    suffix = pathlib.Path(path).suffix

    if suffix == "csv":
        loader = pd.read_csv
    elif suffix in ("dat", "txt"):
        loader = np.loadtxt
    elif suffix in ("npy", "npz"):
        loader = np.load
    elif suffix in ("xyz", "pdb", "mol2"):
        loader = load_text
    else:
        raise ValueError(f"Can't find loader for {suffix} file")

    return loader(path)


def save_data(data: Data, path: Path):
    """
    Parameters
    ----------
    data: numpy.ndarray or pd.DataFrame
    path: pathlib.Path or str
        Filename
    """
    suffix = pathlib.Path(path).suffix

    if suffix == "csv":
        data.to_csv(path, index=False)
    elif suffix in ("dat", "txt"):
        np.savetxt(path, data)
    elif suffix == "npy":
        np.save(path, data)
    elif suffix == "npz":
        np.savez(path, **data)
    elif suffix == "xyz":
        if isinstance(data, str):
            with open(path, "w") as f:
                f.write(data)
        else:
            data.save_xyz_file(path, True)
    else:
        raise ValueError(f"Can't find saver for {suffix} file")
    logger.info(f"Saved to {os.path.abspath(path)}")


class IOMixin(base.Model, abc.ABC):
    """I/O options

    Parameters
    ----------
    save_output: bool
        Whether to save output QM or intermediate files.
    load_input: bool
        Whether to read QM and intermediate files in, where available.
    """
    save_output: bool = False
    load_input: bool = False

    @property
    @abc.abstractmethod
    def path(self):
        raise NotImplementedError

    @contextlib.contextmanager
    def directory(self):
        """Return associated directory.

        This can be used as a context manager.
        If ``save_output`` is False, this is a temporary directory.
        """
        cwd = pathlib.Path.cwd()
        if self.save_output:
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

    def try_load_data(self, path: Path) -> Optional[Data]:
        """Try to load data from given path

        This only tries to load data if `self.load_input = True`.
        If loading the data fails for any reason, the exception is logged and
        None is returned. The path is also returned, whether or not
        the data is loaded.

        Parameters
        ----------
        path: pathlib.Path or str
            Data path

        Returns
        -------
        data: None, numpy.ndarray, or pd.DataFrame
            numpy.ndarray or pd.DataFrame if data is successfully loaded;
            None if not.
        """
        if self.load_input:
            try:
                data = load_data(path)
            except Exception as e:
                logger.error(e)
                logger.info(f"Could not load data from {path}.")
            else:
                logger.info(f"Loaded from {path}.")
                return data

    def try_save_data(self, data: Data, path: Path):
        """Save data to given path

        This only happens if `self.save_output = True`.

        Parameters
        ----------
        data: numpy.ndarray or pd.DataFrame
        path: pathlib.Path or str
            Filename        
        """
        if self.save_output:
            return save_data(data, path)

    def try_datafile(self, path: Path, func, *args, **kwargs) -> Data:
        """Try to load function output from a given file.
        Run the function if this fails, and save to the file.

        Parameters
        ----------
        path: pathlib.Path or str
            Filename
        func: callable
            Function to call
        *args:
            Arguments to pass to ``func``
        **kwargs:
            Keyword arguments to pass to ``func``

        Returns
        -------
        data
        """
        data = self.try_load_data(path)
        if data is not None:
            return data
        data = func(*args, **kwargs)
        self.try_save_data(data, path)
        return data
