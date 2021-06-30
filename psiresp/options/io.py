import logging
import pathlib
import os
import tempfile
import contextlib
from typing import Union, Optional

import pandas as pd
import numpy as np
from numpy.typing import ArrayLike

from .base import options, OptionsBase
from . import utils

logger = logging.getLogger(__name__)




Data = Union[ArrayLike, pd.DataFrame]
Path = Union[pathlib.Path, str]



@options
class IOOptions(OptionsBase):
    """I/O options
    
    Parameters
    ----------
    name: str
        Name of this orientation, conformer, molecule or job
    save_output: bool
        Whether to save output QM or intermediate files.
    load_input: bool
        Whether to read QM and intermediate files in, where available.
    """
    save_output: bool = False
    load_input: bool = False

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
            

    def try_datafile(self, path: Path, func, *args, **kwargs):
        data = self.try_load_data(path)
        if data is not None:
            return data
        data = func(*args, **kwargs)
        self.try_save_data(data, path)
        return data

        
