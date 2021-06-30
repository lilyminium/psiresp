import logging
import pathlib
import os
import tempfile
import abc
import contextlib
from typing import Union, Optional

import pandas as pd
import numpy as np
from numpy.typing import ArrayLike

from . import utils
from ..base import Model

logger = logging.getLogger(__name__)

def datafile(func: Optional[Callable] = None,
             path: Optional[str] = None,
             ) -> Callable:
    """Try to load data from file. If not found, saves data to same path"""

    if func is None:
        return functools.partial(datafile, path=path)
    
    if path is None:
        fname = func.__name__
        if fname.startswith("compute_"):
            fname = fname.split("compute_", maxsplit=1)[1]
        path = fname + ".dat"

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        return self.try_datafile(path, wrapper, self, *args, **kwargs)
    return wrapper


class IOMixin(Model, abc.ABC):
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
                data = utils.load_data(path)
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
            return utils.save_data(data, path)
            

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

        
