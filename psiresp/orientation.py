import os
from dataclasses import dataclass, field, Field

import logging
import subprocess

import numpy as np

from . import base, utils, constants, mixins
from .options import IOOptions

log = logging.getLogger(__name__)


@dataclass
class Orientation(mixins.ContainsParentMixin, base.MoleculeBase):

    conformer: "Conformer"

    def __post_init__(self):
        self._grid = None
        self._esp = None
        self._r_inv = None

    @property
    def _parent(self):
        return self.conformer

    @property
    def qm_options(self):
        return self.conformer.qm_options
    
    @property
    def grid_options(self):
        return self.conformer.grid_options

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


    def compute_r_inv(self)  -> npt.NDArray:
        """Get inverse r"""
        points = self.grid.reshape((len(self.grid), 1, 3))
        disp = self.coordinates - points
        inverse = 1 / np.sqrt(np.einsum("ijk, ijk->ij", disp, disp))
        return inverse * constants.BOHR_TO_ANGSTROM

    def get_esp_mat_a(self) -> npt.NDArray:
        """Get A matrix for solving"""
        return np.einsum("ij, ik->jk", self.r_inv, self.r_inv)

    def get_esp_mat_b(self) -> npt.NDArray:
        """Get B matrix for solving"""
        return np.einsum("i, ij->j", self.esp, self.r_inv)

    @base.datafile(filename="grid.dat")
    def compute_grid(self):
        return self.grid_options.generate_vdw_grid(self.symbols,
                                                   self.coordinates)

    @base.datafile(filename="grid_esp.dat")
    def compute_esp(self):
        assert self.grid is not None
        with self.directory() as tmpdir:
            # ... this dies unless you write out grid.dat
            np.savetxt("grid.dat", self.grid)
            infile = self.qm_options.write_esp_file(self.psi4mol)
            self.qm_options.try_run_qm(infile, cwd=tmpdir)
            esp = np.loadtxt("grid_esp.dat")
        self._esp = esp
        return esp
