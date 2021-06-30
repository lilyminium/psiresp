from __future__ import division, absolute_import
import os
import logging
import subprocess

import numpy as np

from . import base, utils
from .options import IOOptions

log = logging.getLogger(__name__)


class Orientation(base.Psi4MolContainerMixin, base.IOBase):
    """
    Class to manage one orientation of a conformer. This should
    not usually be created or interacted with by a user. Instead,
    users are expected to work primarily with
    :class:`psiresp.conformer.Conformer` or :class:`psiresp.resp.Resp`.

    Parameters
    ----------
    psi4mol: psi4.core.Molecule
        Psi4 molecule that forms the basis of this orientation
    conformer: psiresp.Conformer
        The conformer that owns this orientation
    name: str (optional)
        The name of this orientation
    io_options: psiresp.IOOptions
        input/output options

    Attributes
    ----------
    psi4mol: psi4.core.Molecule
        Psi4 molecule that forms the basis of this orientation
    conformer: psiresp.Conformer
        The conformer that owns this orientation
    grid: numpy.ndarray
        The grid of points on which to compute the ESP
    esp: numpy.ndarray
        The computed ESP at grid points
    r_inv: numpy.ndarray
        inverse distance from each grid point to each atom, in
        atomic units. 
    directory: str
        Directory in which to run Psi4 jobs in
    qm_options:
        QM options with which to run Psi4 jobs. This is actually
        attached to the owning conformer.
    name: str (optional)
        The name of this orientation
    io_options: psiresp.IOOptions
        input/output options
    """
    def __init__(self, psi4mol, conformer, name=None, io_options=IOOptions()):
        super().__init__(psi4mol, name=name, io_options=io_options)
        self.conformer = conformer
        self._grid = None
        self._esp = None
        self._r_inv = None

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

    def compute_r_inv(self):
        points = self.grid.reshape((len(self.grid), 1, 3))
        disp = self.coordinates - points
        inverse = 1 / np.sqrt(np.einsum("ijk, ijk->ij", disp, disp))
        return inverse * self.BOHR_TO_ANGSTROM

    def clone(self, name=None):
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
        new = type(self)(mol, conformer=self.conformer, io_options=self.io_options)
        return new

    def get_esp_mat_a(self):
        """Get A matrix for solving"""
        return np.einsum("ij, ik->jk", self.r_inv, self.r_inv)

    def get_esp_mat_b(self):
        """Get B matrix for solving"""
        return np.einsum("i, ij->j", self.esp, self.r_inv)

    @base.datafile(filename="grid.dat")
    def compute_grid(self):
        return utils.compute_grid(self.conformer.vdw_points,
                                  self.coordinates,
                                  rmin=self.conformer.esp_options.rmin,
                                  rmax=self.conformer.esp_options.rmax)

    @base.datafile(filename="grid_esp.dat")
    def compute_esp(self):
        import psi4
        
        # ... this dies unless you write out grid.dat
        with self.get_subfolder() as tmpdir:
            np.savetxt(os.path.join(tmpdir, "grid.dat"), self.grid)
            infile = f"{self.name}_esp.in"
            outfile = self.qm_options.write_esp_file(self.psi4mol, destination_dir=tmpdir,
                                                     filename=infile)

            # don't use Psi4 API because we need different processes for parallel jobs
            # maybe it's already run?
            if not self.io_options.force and os.path.isfile(outfile):
                try:
                    return np.loadtxt(outfile)
                except:
                    pass
            cmd = f"{psi4.executable} -i {infile}"
            subprocess.run(cmd, shell=True, cwd=tmpdir, stderr=subprocess.PIPE)
            esp = np.loadtxt(outfile)

        return esp
