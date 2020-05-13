from __future__ import division, absolute_import
import os
import logging
import tempfile
import textwrap

import numpy as np

from . import utils

log = logging.getLogger(__name__)

#: convert bohr to angstrom
BOHR_TO_ANGSTROM = 0.52917721092


class Orientation(object):
    """
    Class to manage one Psi4 molecule.

    Parameters
    ----------
    molecule: Psi4 molecule
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

    def __init__(self, molecule, symbols=None, grid_name='grid.dat',
                 esp_name='grid_esp.dat', load_files=False):
        self.name = molecule.name()
        self.n_atoms = molecule.natom()
        if symbols is None:
            symbols = np.array([molecule.symbol(i) for i in range(self.n_atoms)])
        self.symbols = symbols
        self.bohr = 'Bohr' in str(molecule.units())
        self.indices = np.arange(self.n_atoms).astype(int)
        self.molecule = molecule
        self._grid_name = grid_name
        self._esp_name = esp_name
        self.grid_filename = self._prepend_name_to_file(grid_name)
        self.esp_filename = self._prepend_name_to_file(esp_name)
        self._grid = self._esp = self._r_inv = None

        # try to read from files
        if load_files:
            try:
                self.grid = np.loadtxt(self.grid_filename)
            except OSError:
                warnings.warn('Could not read data from {}'.format(self.grid_filename))
            else:
                if self.bohr:
                    self.grid *= BOHR_TO_ANGSTROM
                log.info('Read grid from {}'.format(self.grid_filename))
            try:
                self.esp = np.loadtxt(self.esp_filename)
            except OSError:
                warnings.warn('Could not read data from {}'.format(self.esp_filename))
            else:
                log.info('Read esp from {}'.format(self.esp_filename))

    def _prepend_name_to_file(self, filename):
        head, tail = os.path.split(filename)
        if head and not head.endswith(r'/'):
            head += '/'
        return '{}{}_{}'.format(head, self.name, tail)

    @property
    def grid(self):
        return self._grid

    @grid.setter
    def grid(self, values):
        # ESP and inverse distance depend on grid; redo
        self._grid = values
        self._esp = None
        self._r_inv = None

    @property
    def esp(self):
        return self._esp

    @esp.setter
    def esp(self, values):
        self._esp = values
        self._r_inv = None

    @property
    def r_inv(self):
        """Inverse distance from each grid point to each atom, in atomic units."""
        if self._r_inv is None and self.grid is not None:
            points = self.grid.reshape((len(self.grid), 1, 3))
            disp = self.coordinates-points
            inverse = 1/np.sqrt(np.einsum('ijk, ijk->ij', disp, disp))
            self._r_inv = inverse*BOHR_TO_ANGSTROM

        return self._r_inv

    @property
    def coordinates(self):
        coords = self.molecule.geometry().np.astype('float')*BOHR_TO_ANGSTROM
        return coords

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
        mol = self.molecule.clone()
        if name is not None:
            mol.set_name(name)
        new = type(self)(name, symbols=self.symbols,
                         grid_name=self._grid_name,
                         esp_name=self._esp_name)
        return new

    def get_esp_matrices(self, vdw_points=None, rmin=0, rmax=-1,
                         basis='6-31g*', method='scf', solvent=None,
                         vdw_radii={}, use_radii='msk',
                         vdw_point_density=1.0,
                         vdw_scale_factors=(1.4, 1.6, 1.8, 2.0),
                         psi4_options={}, save_files=False):
        """
        Get A and B matrices to solve for charges, from Bayly:93:Eq.11:: Aq=B

        Parameters
        ----------
        vdw_points: list of tuples (optional)
            List of tuples of (unit_shell_coordinates, scaled_atom_radii).
            If this is ``None`` or empty, new points are generated from 
            utils.gen_connolly_shells and the variables ``vdw_radii``,
            ``use_radii``, ``vdw_scale_factors``, ``vdw_point_density``.
        use_radii: str (optional)
            which set of van der Waals' radii to use. 
            Ignored if ``vdw_points`` is provided.
        vdw_scale_factors: iterable of floats (optional)
            scale factors. Ignored if ``vdw_points`` is provided.
        vdw_point_density: float (optional)
            point density. Ignored if ``vdw_points`` is provided.
        vdw_radii: dict (optional)
            van der Waals' radii. If elements in the molecule are not
            defined in the chosen ``use_radii`` set, they must be given here.
            Ignored if ``vdw_points`` is provided.
        rmin: float (optional)
            inner boundary of shell to keep grid points from
        rmax: float (optional)
            outer boundary of shell to keep grid points from. If < 0,
            all points are selected.
        basis: str (optional)
            Basis set to compute ESP
        method: str (optional)
            Method to compute ESP
        solvent: str (optional)
            Solvent for computing in implicit solvent
        psi4_options: dict (optional)
            additional Psi4 options
        save_files: bool (optional)
            if ``True``, Psi4 files are saved and the computed ESP
            and grids are written to files.

        Returns
        -------
        a: ndarray
        b: ndarray
        """

        if self.grid is None:
            self.get_grid(vdw_points=vdw_points, rmin=rmin,
                          rmax=rmax, vdw_radii=vdw_radii,
                          use_radii=use_radii, density=vdw_point_density,
                          scale_factors=vdw_scale_factors,
                          save_files=save_files)

        if self.esp is None:
            self.get_esp(basis=basis, method=method,
                         psi4_options=psi4_options,
                         save_files=save_files, solvent=solvent)

        a = np.einsum('ij, ik->jk', self.r_inv, self.r_inv)
        b = np.einsum('i, ij->j', self.esp, self.r_inv)

        return a, b

    def get_grid(self, vdw_points=None, rmin=0, rmax=-1, fmt='%15.10f',
                 save_files=False, vdw_radii={}, use_radii='msk',
                 scale_factors=(1.4, 1.6, 1.8, 2.0),
                 density=1.0):
        """
        Get ESP grid from a given file or compute it from the
        van der Waals' surfaces.

        Parameters
        ----------
        vdw_points: list of tuples (optional)
            List of tuples of (unit_shell_coordinates, scaled_atom_radii).
            If this is ``None`` or empty, new points are generated from 
            utils.gen_connolly_shells and the variables ``vdw_radii``,
            ``use_radii``, ``scale_factors``, ``density``.
        rmin: float (optional)
            inner boundary of shell to keep grid points from
        rmax: float (optional)
            outer boundary of shell to keep grid points from. If < 0,
            all points are selected.
        use_radii: str (optional)
            which set of van der Waals' radii to use. 
            Ignored if ``vdw_points`` is provided.
        scale_factors: iterable of floats (optional)
            scale factors. Ignored if ``vdw_points`` is provided.
        density: float (optional)
            point density. Ignored if ``vdw_points`` is provided.
        vdw_radii: dict (optional)
            van der Waals' radii. If elements in the molecule are not
            defined in the chosen ``use_radii`` set, they must be given here.
            Ignored if ``vdw_points`` is provided.
        fmt: str (optional)
            float format
        save_files: bool (optional)
            if ``True``, Psi4 files are saved and the computed grid
            is written to a file.

        Returns
        -------
        grid: ndarray
        """
        # usually generated in RESP or Conformer
        if vdw_points is None or len(vdw_points) == 0:
            vdw_points = utils.gen_connolly_shells(self.symbols,
                                                   vdw_radii=vdw_radii,
                                                   use_radii=use_radii,
                                                   scale_factors=scale_factors,
                                                   density=density)
        points = []
        for pts, rad in vdw_points:
            points.append(utils.gen_vdw_surface(pts, rad, self.coordinates,
                                                rmin=rmin, rmax=rmax))
        points = to_save = np.concatenate(points)

        if save_files:
            if self.bohr:
                to_save = points/BOHR_TO_ANGSTROM
            np.savetxt(self.grid_filename, to_save, fmt=fmt)

        self.grid = points
        log.debug('Computed grid for {} with {} points'.format(self.name,
                                                               len(points)))
        return points

    def get_esp(self, basis='6-31g*', method='scf', solvent=None,
                psi4_options={}, fmt='%15.10f', save_files=False):
        """
        Get ESP at each point on a grid from a given file or compute it with Psi4.

        Parameters
        ----------
        basis: str (optional)
            Basis set to compute ESP
        method: str (optional)
            Method to compute ESP
        solvent: str (optional)
            Solvent for computing in implicit solvent
        psi4_options: dict (optional)
            additional Psi4 options
        fmt: str
            float format
        save_files: bool (optional)
            if ``True``, Psi4 files are saved and the computed ESP 
            is written to a file.

        Returns
        -------
        grid_esp: ndarray
        """
        if not save_files:
            cwd = os.getcwd()
            with tempfile.TemporaryDirectory(prefix='tmp') as tmpdir:
                os.chdir(tmpdir)
                self._get_esp(method=method, basis=basis,
                              psi4_options=psi4_options, solvent=solvent,
                              fmt=fmt)
                os.chdir(cwd)
        else:
            self._get_esp(method=method, basis=basis,
                          psi4_options=psi4_options, solvent=solvent,
                          fmt=fmt)
        return self.esp

    def _get_esp(self, basis='6-31g*', method='scf', solvent=None,
                 psi4_options={}, fmt='%15.10f'):
        """
        Get ESP at each point on a grid from a given file or compute it with Psi4.

        Parameters
        ----------
        basis: str (optional)
            Basis set to compute ESP
        method: str (optional)
            Method to compute ESP
        solvent: str (optional)
            Solvent for computing in implicit solvent
        psi4_options: dict (optional)
            additional Psi4 options
        fmt: str
            float format

        Returns
        -------
        grid_esp: ndarray
        """
        import psi4
        grid_fn, esp_fn = 'grid.dat', 'grid_esp.dat'
        logfile = self.esp_filename.rsplit('.', maxsplit=1)[0]+'.log'
        if self.bohr:  # I THINK psi4 converts units based on molecule units
            np.savetxt(grid_fn, self.grid/BOHR_TO_ANGSTROM, fmt=fmt)
        else:
            np.savetxt(grid_fn, self.grid, fmt=fmt)

        psi4.set_output_file(logfile)
        psi4.set_options({'basis': basis})
        psi4.set_options(psi4_options)

        msg = 'Computing grid ESP for {} with {}/{}, solvent={}'
        log.debug(msg.format(self.name, method, basis, solvent))

        if solvent:
            psi4.set_options({'pcm': True,
                              'pcm_scf_type': 'total'})
            fname = psi4.core.get_local_option('PCM', 'PCMSOLVER_PARSED_FNAME')
            if not fname or not os.path.exists(fname):
                block = textwrap.dedent("""
                    units = angstrom
                    medium {{
                        solvertype = CPCM
                        solvent = {}
                    }}
                    cavity {{
                        radiiset = bondi  # Bondi | UFF | Allinger
                        type = gepol
                        scaling = True  # radii for spheres scaled by 1.2
                        area = 0.3
                        mode = implicit
                    }}
                """.format(solvent))
                psi4.pcm_helper(block)

        E, wfn = psi4.prop(method, properties=['GRID_ESP'], molecule=self.molecule,
                           return_wfn=True)

        if solvent:
            import pcmsolver  # clear pcmsolver or it's sad the next time
            pcmsolver.getkw.GetkwParser.bnf = None

        self.esp = np.array(wfn.oeprop.Vvals())
        psi4.core.clean()
        psi4.core.clean_options()
        np.savetxt(self.esp_filename, self.esp, fmt=fmt)
        return self.esp
