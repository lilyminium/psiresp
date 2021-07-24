from typing import Optional
import pathlib

import numpy as np
from pydantic import PrivateAttr

from . import mixins, utils
from .utils.io import datafile


class BaseMoleculeChild(mixins.MoleculeMixin, mixins.ContainsQMandGridOptions):
    _parent_path: pathlib.Path = PrivateAttr(default=".")

    @property
    def default_path(self):
        return pathlib.Path(self._parent_path) / self.name


class Orientation(BaseMoleculeChild, mixins.OrientationOptions):
    """
    Class to manage one orientation of a conformer. This should
    not usually be created or interacted with by a user. Instead,
    users are expected to work primarily with
    :class:`psiresp.conformer.Conformer` or :class:`psiresp.resp.Resp`.

    Attributes
    ----------
    grid: numpy.ndarray
        The grid of points on which to compute the ESP
    esp: numpy.ndarray
        The computed ESP at grid points
    r_inv: numpy.ndarray
        inverse distance from each grid point to each atom, in
        atomic units.
    """

    # conformer: mixins.ConformerOptions = Field(description="The conformer that owns this orientation")
    _grid: Optional[np.ndarray] = PrivateAttr(default=None)
    _esp: Optional[np.ndarray] = PrivateAttr(default=None)
    _r_inv: Optional[np.ndarray] = PrivateAttr(default=None)

    @property
    def grid(self):
        if self._grid is None:
            self._grid = self.compute_grid()
        return self._grid

    @property
    def esp(self):
        if self._esp is None:
            self._esp = self.compute_esp()
        return self._esp

    @property
    def r_inv(self):
        if self._r_inv is None:
            self._r_inv = self.compute_r_inv()
        return self._r_inv

    def compute_r_inv(self) -> np.ndarray:
        """Get inverse r"""
        points = self.grid.reshape((len(self.grid), 1, 3))
        disp = self.coordinates - points
        inverse = 1 / np.sqrt(np.einsum("ijk, ijk->ij", disp, disp))
        return inverse * utils.BOHR_TO_ANGSTROM

    def get_esp_mat_a(self) -> np.ndarray:
        """Get A matrix for solving"""
        return np.einsum("ij, ik->jk", self.r_inv, self.r_inv)

    def get_esp_mat_b(self) -> np.ndarray:
        """Get B matrix for solving"""
        return np.einsum("i, ij->j", self.esp, self.r_inv)

    @datafile(filename="{self.name}_grid.dat")
    def compute_grid(self, grid_options=None):
        if grid_options is None:
            grid_options = self.grid_options
        grid = grid_options.generate_vdw_grid(self.symbols, self.coordinates)
        self._grid = grid
        return grid

    @datafile(filename="{self.name}_grid_esp.dat")
    def compute_esp(self, qm_options=None):
        if qm_options is None:
            qm_options = self.qm_options
        grid = self.grid
        if self.psi4mol_geometry_in_bohr:
            grid = grid * utils.ANGSTROM_TO_BOHR
        with self.directory() as tmpdir:
            # ... this dies unless you write out grid.dat
            np.savetxt("grid.dat", grid)
            infile = qm_options.write_esp_file(self.psi4mol,
                                               name=self.name)
            proc = qm_options.try_run_qm(infile, cwd=tmpdir)
            esp = np.loadtxt("grid_esp.dat")
        self._esp = esp
        # assert len(self._esp)
        return esp
