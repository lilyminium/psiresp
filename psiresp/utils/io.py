
import logging
import pathlib
import os
import functools
from typing import Union, Optional, Callable

import pandas as pd
import numpy as np
from numpy.typing import ArrayLike

logger = logging.getLogger(__name__)

Data = Union[ArrayLike, pd.DataFrame]
Path = Union[pathlib.Path, str]


def datafile(func: Optional[Callable] = None,
             filename: Optional[str] = None,
             ) -> Callable:
    """Try to load data from file. If not found, saves data to same path"""

    if func is None:
        return functools.partial(datafile, filename=filename)
    elif isinstance(func, str):
        return functools.partial(datafile, filename=func)

    if filename is None:
        fname = func.__name__
        if fname.startswith("compute_"):
            fname = fname.split("compute_", maxsplit=1)[1]
        filename = fname + ".dat"

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        return self.try_datafile(filename, func, self, *args, **kwargs)
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
    suffix = pathlib.Path(path).suffix.strip(".")

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
    suffix = pathlib.Path(path).suffix.strip(".")

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
