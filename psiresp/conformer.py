from __future__ import absolute_import
import logging
import os
import tempfile

import numpy as np

from . import utils
from .orientation import Orientation

log = logging.getLogger(__name__)


class Conformer(object):
    """
    Wrapper class to manage one conformer molecule, containing 
    multiple orientations.

    Parameters
    ----------
    molecule: Psi4 molecule
    charge: int (optional)
        overall charge of the molecule.
    multiplicity: int (optional)
        multiplicity of the molecule
    name: str (optional)
        name of the molecule. This is used to name output files. If not 
        given, the default Psi4 name is 'default'.
    orient: list of tuples of ints (optional)
        List of reorientations. Corresponds to REMARK REORIENT in R.E.D.
        e.g. [(1, 5, 9), (9, 5, 1)] creates two reorientations: the first
        around the first, fifth and ninth atom; and the second in reverse 
        order.
    rotate: list of tuples of ints (optional)
        List of rotations. Corresponds to REMARK ROTATE in R.E.D.
        e.g. [(1, 5, 9), (9, 5, 1)] creates two rotations: the first
        around the first, fifth and ninth atom; and the second in reverse 
        order.
    translate: list of tuples of floats (optional)
        List of translations. Corresponds to REMARK TRANSLATE in R.E.D.
        e.g. [(1.0, 0, -0.5)] creates a translation that adds 1.0 to the 
        x coordinates, 0 to the y coordinates, and -0.5 to the z coordinates.
    load_files: bool (optional)
        If ``True``, tries to load ESP and grid data from file.

    Attributes
    ----------
    molecule: Psi4 molecule
    name: str
        name of the molecule. This is used to name output files.
    n_atoms: int
        number of atoms in each conformer
    symbols: ndarray
        element symbols
    orientations: list of Orientations
        list of the molecule with reoriented coordinates
    """

    def __init__(self, molecule, charge=0, multiplicity=1, name=None,
                 orient=[], rotate=[], translate=[], load_files=False):
        if name and name != molecule.name():
            molecule.set_name(name)
        self.name = molecule.name()
        if charge != molecule.molecular_charge():
            molecule.set_molecular_charge(charge)
        if multiplicity != molecule.multiplicity():
            molecule.set_multiplicity(multiplicity)
        self.charge = charge
        self.multiplicity = multiplicity
        self.molecule = molecule
        self.n_atoms = molecule.natom()
        self.symbols = np.array([molecule.symbol(i) for i in range(self.n_atoms)])
        self._orient = orient[:]
        self._rotate = rotate[:]
        self._translate = translate[:]
        self._load_files = load_files
        self._orientations = []
        self._orientation = Orientation(self.molecule, symbols=self.symbols)

    @property
    def n_orientations(self):
        return len(self.orientations)

    @property
    def orientations(self):
        if not self._orientations:
            self._gen_orientations(orient=self._orient,
                                   translate=self._translate,
                                   rotate=self._rotate,
                                   load_files=self._load_files)
        if not self._orientations:
            return [self._orientation]
        return self._orientations

    @property
    def _grid_needs_computing(self):
        return any(mol.grid is None for mol in self.orientations)

    @property
    def coordinates(self):
        return self.molecule.geometry().np.astype('float')

    def clone(self, name=None):
        """Clone into another instance of Conformer

        Parameters
        ----------
        name: str (optional)
            If not given, the new Conformer instance has '_copy' appended 
            to its name

        Returns
        -------
        Conformer
        """
        if name is None:
            name = self.name+'_copy'
        charge = self.charge
        mult = self.multiplicity
        new = type(self)(self.molecule.clone(), name=name, charge=charge,
                         mult=mult, orient=self._orient, rotate=self._rotate,
                         translate=self._translate,
                         load_files=self._load_files)
        return new

    def _gen_orientations(self, orient=[], translate=[], rotate=[],
                          load_files=False):
        """
        Generate new orientations.

        Parameters
        ----------
        orient: list of tuples of ints (optional)
            List of reorientations. Corresponds to REMARK REORIENT in R.E.D.
            e.g. [(1, 5, 9), (9, 5, 1)] creates two reorientations: the first
            around the first, fifth and ninth atom; and the second in reverse 
            order.
        rotate: list of tuples of ints (optional)
            List of rotations. Corresponds to REMARK ROTATE in R.E.D.
            e.g. [(1, 5, 9), (9, 5, 1)] creates two rotations: the first
            around the first, fifth and ninth atom; and the second in reverse 
            order.
        translate: list of tuples of floats (optional)
            List of translations. Corresponds to REMARK TRANSLATE in R.E.D.
            e.g. [(1.0, 0, -0.5)] creates a translation that adds 1.0 to the 
            x coordinates, 0 to the y coordinates, and -0.5 to the z coordinates.
        load_files: bool (optional)
            If ``True``, each orientation tries to load ESP and grid data from file.
        """

        for atom_ids in orient:
            a, b, c = [a-1 if a > 0 else a for a in atom_ids]
            xyz = utils.orient_rigid(a, b, c, self.coordinates)
            self._add_orientation(xyz, load_files=load_files)

        for atom_ids in rotate:
            a, b, c = [a-1 if a > 0 else a for a in atom_ids]
            xyz = utils.rotate_rigid(a, b, c, self.coordinates)
            self._add_orientation(xyz, load_files=load_files)

        for translation in translate:
            xyz = self.coordinates+translation
            self._add_orientation(xyz, load_files=load_files)

    def _add_orientation(self, coordinates, load_files=False):
        import psi4
        mat = psi4.core.Matrix.from_array(coordinates)
        cmol = self.molecule.clone()
        cmol.set_geometry(mat)
        cmol.fix_com(True)
        cmol.fix_orientation(True)
        cmol.update_geometry()
        cmol.set_name('{}_o{}'.format(self.name, len(self._orientations)+1))
        self._orientations.append(Orientation(cmol, symbols=self.symbols,
                                              load_files=load_files))

    def add_orientations(self, orient=[], translate=[], rotate=[],
                         load_files=False):
        """
        Add new orientations to generate.

        Parameters
        ----------
        orient: list of tuples of ints (optional)
            List of reorientations. Corresponds to REMARK REORIENT in R.E.D.
            e.g. [(1, 5, 9), (9, 5, 1)] creates two reorientations: the first
            around the first, fifth and ninth atom; and the second in reverse 
            order.
        rotate: list of tuples of ints (optional)
            List of rotations. Corresponds to REMARK ROTATE in R.E.D.
            e.g. [(1, 5, 9), (9, 5, 1)] creates two rotations: the first
            around the first, fifth and ninth atom; and the second in reverse 
            order.
        translate: list of tuples of floats (optional)
            List of translations. Corresponds to REMARK TRANSLATE in R.E.D.
            e.g. [(1.0, 0, -0.5)] creates a translation that adds 1.0 to the 
            x coordinates, 0 to the y coordinates, and -0.5 to the z coordinates.
        load_files: bool (optional)
            If ``True``, each orientation tries to load ESP and grid data from file.
        """
        self._orient.extend(orient)
        self._translate.extend(translate)
        self._rotate.extend(rotate)
        if self._orientations:
            self._gen_orientations(orient=orient, rotate=rotate,
                                   translate=translate, load_files=load_files)

    def optimize_geometry(self, method='scf', basis='6-31g*',
                          psi4_options={}, save_opt_geometry=True,
                          save_files=False):
        """
        Optimise the geometry of the molecule and update the coordinates.

        Parameters
        ----------
        multiplicity: int (optional)
        basis: str (optional)
            Basis set to optimise geometry
        method: str (optional)
            Method to optimise geometry
        psi4_options: dict (optional)
            additional Psi4 options
        save_opt_geometry: bool (optional)
            if ``True``, writes the optimised geometry to an XYZ file.
        save_files: bool (optional)
            if ``True``, Psi4 files are saved. This does not affect 
            writing optimised geometries to files. 
        """
        if not save_files:
            cwd = os.getcwd()
            with tempfile.TemporaryDirectory(prefix='tmp') as tmpdir:
                os.chdir(tmpdir)
                self._optimize_geometry(method=method, basis=basis,
                                        psi4_options=psi4_options)
                os.chdir(cwd)
        else:
            self._optimize_geometry(method=method, basis=basis,
                                    psi4_options=psi4_options)

        if save_opt_geometry:
            self.molecule.save_xyz_file(self.name+'_opt.xyz', True)

    def _optimize_geometry(self, method='scf', basis='6-31g*', psi4_options={}):
        import psi4
        psi4.set_options({'basis': basis,
                          'geom_maxiter': 200,
                          'full_hess_every': 10,
                          'g_convergence': 'gau_tight',
                          })
        psi4.set_options(psi4_options)
        logfile = self.name+'_opt.log'
        psi4.set_output_file(logfile, False)  # doesn't work? where is the output?!
        e = psi4.optimize(method, molecule=self.molecule)
        psi4.core.clean()
        self._orientations = []

    def get_esp_matrices(self, weight=1.0, vdw_points=None, rmin=0, rmax=-1,
                         basis='6-31g*', method='scf', solvent=None,
                         psi4_options={}, save_files=False, vdw_radii={},
                         use_radii='msk', vdw_point_density=1.0,
                         vdw_scale_factors=(1.4, 1.6, 1.8, 2.0)):
        """
        Get A and B matrices to solve for charges, from Bayly:93:Eq.11:: Aq=B

        Parameters
        ----------
        weight: float (optional)
            how much to weight the matrices
        vdw_points: list of tuples (optional)
            List of tuples of (unit_shell_coordinates, scaled_atom_radii).
            If this is ``None`` or empty, new points are generated from 
            utils.gen_connolly_shells and the variables ``vdw_radii``,
            ``use_radii``, ``vdw_scale_factors``, ``vdw_point_density``.
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
        psi4_options: dict (optional)
            additional Psi4 options
        save_files: bool (optional)
            if ``True``, Psi4 files are saved and the computed ESP
            and grids are written to files.

        Returns
        -------
        A: ndarray (n_atoms, n_atoms)
        B: ndarray (n_atoms,)
        """
        A = np.zeros((self.n_atoms, self.n_atoms))
        B = np.zeros(self.n_atoms)

        for mol in self.orientations:
            a, b = mol.get_esp_matrices(vdw_points=vdw_points,
                                        basis=basis, method=method,
                                        solvent=solvent,
                                        psi4_options=psi4_options,
                                        save_files=save_files)
            A += a
            B += b

        A *= (weight**2)
        B *= (weight**2)

        return A, B
