from __future__ import division, absolute_import
import os
import logging
import tempfile
import textwrap
import warnings
import subprocess

import numpy as np

from . import base, utils

log = logging.getLogger(__name__)

#: convert bohr to angstrom
BOHR_TO_ANGSTROM = 0.52917721092


class Orientation(base.CachedBase):
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

    kwargnames = ["method", "basis", "vdw_radii", "symbols", "name",
                  "rmin", "rmax", "use_radii", "scale_factors",
                  "density", "solvent", "psi4_options"]

    def __init__(self, molecule, conformer=None,
                 method="scf", solvent=None,
                 basis="6-31g*", name=None,
                 force=False, verbose=False,
                 n_atoms = None, symbols=None,
                 rmin=0, rmax=-1, use_radii="msk",
                 vdw_radii={},
                 scale_factors=[1.4, 1.6, 1.8, 2.0],
                 density=1.0, psi4_options={},
                 run_qm=True, **kwargs):
        super().__init__(force=force, verbose=verbose, name=name)
        if name is not None:
            molecule.set_name(name)
        self.name = molecule.name()
        self.conformer = conformer

        self.n_atoms = molecule.natom()
        if symbols is None:
            symbols = np.array([molecule.symbol(i) for i in range(self.n_atoms)])
        self.symbols = symbols
        self.bohr = 'Bohr' in str(molecule.units())
        self.indices = np.arange(self.n_atoms).astype(int)
        self.molecule = molecule
        self.method = method
        self.solvent = solvent
        self.psi4_options = dict(**psi4_options)
        self.psi4_options["basis"] = basis
        self.basis = basis
        self.rmin = rmin
        self.rmax = rmax
        self.use_radii = use_radii
        self.scale_factors = scale_factors
        self.density = density
        self.vdw_radii = vdw_radii
        self.run_qm = run_qm

    def __getstate__(self):
        dct = self.kwargs
        for key in ("bohr", "indices", "n_atoms"):
            dct[key] = getattr(self, key)
        dct[utils.PKL_MOLKEY] = utils.psi42xyz(self.molecule)
        dct[utils.PKL_CACHEKEY] = {**self._cache}
        return dct
    
    def __setstate__(self, state):
        mol = state.pop(utils.PKL_MOLKEY)
        cache = state.pop(utils.PKL_CACHEKEY)
        for key, attr in state.items():
            setattr(self, key, attr)
        self._cache = {**cache}
        self.molecule = utils.xyz2psi4(mol)
        self.molecule.set_name(self.name)
        

    @property
    def kwargs(self):
        dct = {}
        for kw in self.kwargnames:
            dct[kw] = getattr(self, kw)
        return dct

    def get_coordinates(self):
        return self.molecule.geometry().np.astype('float')*BOHR_TO_ANGSTROM

    def get_r_inv(self):
        points = self.grid.reshape((len(self.grid), 1, 3))
        disp = self.coordinates - points
        inverse = 1/np.sqrt(np.einsum('ijk, ijk->ij', disp, disp))
        return inverse * BOHR_TO_ANGSTROM


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
        new = type(self)(name, **self.kwargs)
        return new

    def get_esp_mat_a(self):
        return np.einsum('ij, ik->jk', self.r_inv, self.r_inv)
    
    def get_esp_mat_b(self):
        return np.einsum('i, ij->j', self.esp, self.r_inv)

    def get_vdw_points(self):
        return utils.gen_connolly_shells(self.symbols,
                                         vdw_radii=self.vdw_radii,
                                         use_radii=self.use_radii,
                                         scale_factors=self.scale_factors,
                                         density=self.density)

    @utils.datafile
    def get_grid(self):
        points = []
        for pts, rad in self.vdw_points:
            points.append(utils.gen_vdw_surface(pts, rad, self.coordinates,
                                                rmin=self.rmin,
                                                rmax=self.rmax))
        return np.concatenate(points)

    @utils.datafile
    def get_esp(self):
        import psi4

        # ... this dies unless you write out grid.dat
        esp_file = utils.create_psi4_molstr(self.molecule)
        esp_file += f"set basis {self.basis}\n"

        if self.solvent:
            esp_file += textwrap.dedent(f"""
            set {{
                pcm true
                pcm_scf_type total
            }}

            pcm = {{
                Units = Angstrom
                Medium {{
                    SolverType = CPCM
                    Solvent = {self.solvent}
                }}

                Cavity {{
                    RadiiSet = bondi # Bondi | UFF | Allinger
                    Type = GePol
                    Scaling = True # radii for spheres scaled by 1.2
                    Area = 0.3
                    Mode = Implicit
                }}
            }}

            """)
        
        esp_file += textwrap.dedent(f"""
        E, wfn = prop('{self.method}', properties=['GRID_ESP'], return_wfn=True)
        esp = wfn.oeprop.Vvals()
            """)

        try:
            os.mkdir(self.conformer.name)
        except FileExistsError:
            pass
        
        tmpdir = f"{self.conformer.name}/{self.name}_esp"
        try:
            os.mkdir(tmpdir)
        except FileExistsError:
            pass

        np.savetxt(os.path.join(tmpdir, "grid.dat"), self.grid)

        infile = f"{self.name}_esp.in"
        with open(os.path.join(tmpdir, infile), "w") as f:
            f.write(esp_file)
        
        cmd = f"cd {tmpdir}; psi4 -i {infile}; cd -"
        outfile = os.path.join(tmpdir, "grid_esp.dat")
        if self.run_qm:
            subprocess.run(cmd, shell=True)
        elif not os.path.isfile(outfile):
            return cmd

        return np.loadtxt(outfile)
