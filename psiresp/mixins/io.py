import logging
import pathlib
import os
import tempfile
import contextlib
from typing import Union, Optional

from pydantic import Field

from .. import base
from ..utils.io import Data, Path, save_data, load_data

logger = logging.getLogger(__name__)


class IOOptions(base.Model):
    """I/O options"""
    save_output: bool = Field(
        default=False,
        description="Whether to save output QM or intermediate files."
    )
    load_input: bool = Field(
        default=False,
        description="Whether to read available QM and intermediate files in."
    )
    directory_path: Optional[Union[pathlib.Path, str]] = Field(
        default=None,
        description=("Directory path for saving files. "
                     "If not given, defaults to the name of the instance.")
    )


class IOMixin(IOOptions):

    @property
    def path(self):
        return self.directory_path

    @classmethod
    def from_model(cls, obj, **kwargs) -> "Model":
        """Construct an instance from compatible attributes of
        ``object`` and ``kwargs``"""
        new_kwargs = dict(**kwargs)
        new_kwargs["directory_path"] = None
        return super().from_model(obj, **new_kwargs)

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
            os.chdir(path.name)
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
                logger.warning(e)
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
        import numpy as np
        path = str(self.path / path.format(self=self))
        data = self.try_load_data(path)
        if data is not None:
            return data
        data = func(*args, **kwargs)
        self.try_save_data(data, path)
        return data
