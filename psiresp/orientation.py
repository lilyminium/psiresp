from __future__ import division, absolute_import
import os
import logging
import tempfile
import textwrap
import warnings
import subprocess
from typing import Optional, List, Dict

import numpy as np
import psi4

from . import base, utils
from .options import ESPOptions, IOOptions

log = logging.getLogger(__name__)

#: convert bohr to angstrom
BOHR_TO_ANGSTROM = 0.52917721092


class Orientation(base.IOBase, base.Psi4MolContainerMixin):
    """
    Class to manage one Psi4 molecule.

    Parameters
    ----------
    psi4mol: Psi4 molecule
    symbols: list (optional)
        molecule elements. If not given, this is generated from
        the Psi4 molecule.
    grid_name: str (optional)
        template for grid filename. If ``load_files=True``, the
        class tries to load a grid from $molname_$grid_name.
    esp_name: str (optional)
        template for ESP filename. If ``load_files=True``, the
        class tries to load ESP points from $molname_$esp_name.
    load_files: bool (optional)
        If ``True``, tries to load data from file.

    Attributes
    ----------
    molecule: Psi4 molecule
    symbols: ndarray
        molecule elements
    bohr: bool
        if the molecule coordinates are in bohr
    coordinates: ndarray
        molecule coordinates in angstrom
    grid_filename: str
        file to load grid data from, or save it to.
    esp_filename: str
        file to load ESP data from, or save it to.
    grid: ndarray
        grid of points to compute ESP for. Only populated when
        ``get_grid()`` is called.
    esp: ndarray
        ESP at grid points. Only populated when ``get_esp()``
        is called.
    r_inv: ndarray
        inverse distance from each grid point to each atom, in
        atomic units. Only populated when ``get_inverse_distance()``
        is called.
    """

    def __init__(self, psi4mol, conformer, name: Optional[str]=None,
                 io_options=IOOptions()):
        if name is not None:
            psi4mol.set_name(name)
        else:
            name = psi4mol.name()
        super().__init__(name=name, io_options=io_options)

        self.psi4mol = psi4mol
        self.conformer = conformer

        self._grid = None
        self._esp = None
        self._r_inv = None
        self._directory = None

    @property
    def qm_options(self):
        return self.conformer.qm_options

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

    @property
    def directory(self):
        if self._directory is None:
            path = os.path.join(self.conformer.directory, self.name)
            try:
                os.mkdir(path)
            except FileExistsError:
                pass
            self._directory = os.path.abspath(path)
        return self._directory

    # def __getstate__(self):
    #     return dict(psi4mol=utils.psi42xyz(self.psi4mol), name=self.name)

    # def __setstate__(self, state):
    #     self.psi4mol = utils.psi4mol_from_state(state)
    #     self.name = self.psi4mol.name()

    @property
    def coordinates(self):
        return self.psi4mol.geometry().np.astype("float") * BOHR_TO_ANGSTROM

    def compute_r_inv(self):
        points = self.grid.reshape((len(self.grid), 1, 3))
        disp = self.coordinates - points
        inverse = 1 / np.sqrt(np.einsum("ijk, ijk->ij", disp, disp))
        return inverse * BOHR_TO_ANGSTROM

    def clone(self, name: Optional[str]=None):
        """Clone into another instance of Orientation

        Parameters
        ----------
        name: str (optional)
            If not given, the new Resp instance has the same name as this one

        Returns
        -------
        Orientation
        """
        mol = self.psi4mol.clone()
        if name is not None:
            mol.set_name(name)
        new = type(self)(mol, conformer=self.conformer)
        return new

    def get_esp_mat_a(self):
        return np.einsum("ij, ik->jk", self.r_inv, self.r_inv)

    def get_esp_mat_b(self):
        return np.einsum("i, ij->j", self.esp, self.r_inv)

    @base.datafile(filename="grid.dat")
    def compute_grid(self):
        points = []
        for pts, rad in self.conformer.vdw_points:
            surface = utils.gen_vdw_surface(pts, rad, self.coordinates,
                                            rmin=self.conformer.esp_options.rmin,
                                            rmax=self.conformer.esp_options.rmax)
            points.append(surface)
        return np.concatenate(points)

    @base.datafile(filename="esp.dat")
    def compute_esp(self):
        import psi4

        # ... this dies unless you write out grid.dat
        tmpdir = self.directory
        np.savetxt(os.path.join(tmpdir, "grid.dat"), self.grid)

        infile = f"{self.name}_esp.in"

        outfile = self.qm_options.write_esp_file(self.psi4mol,
                                                 destination_dir=tmpdir,
                                                 filename=infile)

        cmd = f"cd {tmpdir}; psi4 -i {infile}; cd -"
        # maybe it's already run?
        if not self.io_options.force and os.path.isfile(outfile):
            return np.loadtxt(outfile)
        subprocess.run(cmd, shell=True)
        return np.loadtxt(outfile)
