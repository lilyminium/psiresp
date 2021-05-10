import logging
import os
import tempfile
import warnings
import pickle
import subprocess
import textwrap
from typing import Optional
from concurrent.futures import as_completed

import numpy as np
import psi4

from . import utils, base
from .orientation import Orientation
# from .type_aliases import Psi4Basis, Psi4Method, AtomReorient, TranslateReorient
from .options import QMOptions, ESPOptions, OrientationOptions, IOOptions

log = logging.getLogger(__name__)


class Conformer(base.IOBase, base.Psi4MolContainerMixin):
    """
    Wrapper class to manage one conformer psi4mol, containing
    multiple orientations.

    Parameters
    ----------
    psi4mol: Psi4 psi4mol
    charge: int (optional)
        overall charge of the psi4mol.
    multiplicity: int (optional)
        multiplicity of the psi4mol
    name: str (optional)
        name of the psi4mol. This is used to name output files. If not
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
    grid_name: str (optional)
            template for grid filename for each Orientation.
    esp_name: str (optional)
        template for ESP filename for each Orientation.
    load_files: bool (optional)
        If ``True``, tries to load ESP and grid data from file.

    Attributes
    ----------
    psi4mol: Psi4 molecule
    name: str
        name of the psi4mol. This is used to name output files.
    n_atoms: int
        number of atoms in each conformer
    symbols: ndarray
        element symbols
    orientations: list of Orientations
        list of the psi4mol with reoriented coordinates
    """

    kwargnames = ("optimize_geometry", "charge", "multiplicity", "weight",
                  "qm_options", "esp_options", "orientation_options",
                  "io_options")

    def __init__(self,
                 psi4mol: psi4.core.Molecule,
                 name: str = "conf",
                 charge: float = 0,
                 multiplicity: int = 1,
                 optimize_geometry: bool = False,
                 weight: float = 1,
                 io_options = IOOptions(),
                 qm_options = QMOptions(),
                 esp_options = ESPOptions(),
                 orientation_options = OrientationOptions()):
        if name and name != psi4mol.name():
            psi4mol.set_name(name)
        super().__init__(name=psi4mol.name(), io_options=io_options)

        self.psi4mol = psi4mol

        # options
        self.optimize_geometry = optimize_geometry
        self.qm_options = qm_options
        self.esp_options = esp_options
        self.orientation_options = orientation_options

        # molecule stuff
        self.charge = charge
        self.multiplicity = multiplicity
        self.weight = weight

        # resp stuff
        self.optimized = False
        self._vdw_points = None
        self._unweighted_ab = None
        self._unweighted_a_matrix = None
        self._unweighted_b_matrix = None
        self._directory = None

        self.post_init()

    @property
    def charge(self):
        return self.psi4mol.molecular_charge()
    
    @charge.setter
    def charge(self, value):
        if value != self.psi4mol.molecular_charge():
            self.psi4mol.set_molecular_charge(value)
            self.psi4mol.update_geometry()

    @property
    def multiplicity(self):
        return self.psi4mol.multiplicity()

    @multiplicity.setter
    def multiplicity(self, value):
        if value != self.psi4mol.multiplicity():
            self.psi4mol.set_multiplicity(value)
            self.psi4mol.update_geometry()


    def post_init(self):
        self._orientations = []
        self.add_orientations()
        self.directory

    @property
    def directory(self):
        if self._directory is None:
            path = os.path.join(os.getcwd(), self.name)
            try:
                os.mkdir(path)
            except FileExistsError:
                pass
            self._directory = path
        return self._directory


    def __getstate__(self):
        dct = dict(psi4mol=utils.psi42xyz(self.psi4mol), name=self.name,
                   optimized=self.optimized, _vdw_points=self._vdw_points,
                   _unweighted_ab=self._unweighted_ab)
        for kwarg in self.kwargnames:
            dct[kwarg] = getattr(self, kwarg)
        return dct

    def __setstate__(self, state):
        self.psi4mol = utils.psi4mol_from_state(state)
        self.name = self.psi4mol.name()
        for kwarg in self.kwargnames:
            setattr(self, kwarg, state[kwarg])
        self.post_init()

    @property
    def coordinates(self):
        return self.psi4mol.geometry().np.astype("float")

    @property
    def qm_options(self):
        return self._qm_options

    @qm_options.setter
    def qm_options(self, options):
        self._qm_options = QMOptions(**options)

    @property
    def esp_options(self):
        return self._esp_options

    @esp_options.setter
    def esp_options(self, options):
        self._esp_options = ESPOptions(**options)

    @property
    def orientation_options(self):
        return self._orientation_options

    @orientation_options.setter
    def orientation_options(self, options):
        self._orientation_options = OrientationOptions(**options)


    @property
    def n_orientations(self):
        return len(self.orientations)

    @property
    def orientations(self):
        return self._orientations
    

    def clone(self, name: Optional[str] = None):
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
            name = self.name + "_copy"

        dct = {}
        for kwarg in self.kwargnames:
            dct[kwarg] = getattr(self, kwarg)

        new = type(self)(self.psi4mol.clone(), name=name, **dct)
        return new

    def _add_orientation(self, coordinates=None):
        import psi4

        cmol = self.psi4mol.clone()
        if coordinates is not None:
            mat = psi4.core.Matrix.from_array(coordinates)
            cmol.set_geometry(mat)
            cmol.set_molecular_charge(self.charge)
            cmol.set_multiplicity(self.multiplicity)
            cmol.fix_com(True)
            cmol.fix_orientation(True)
            cmol.update_geometry()
        name = "{}_o{:03d}".format(self.name, len(self._orientations) + 1)
        cmol.set_name(name)

        omol = Orientation(cmol, conformer=self, name=name)
        self._orientations.append(omol)

    def add_orientations(self):
        self._orientations = []
        if self.orientation_options.keep_original or not self.orientation_options.n_orientations:
            self._add_orientation()  # original
        symbols = [self.psi4mol.symbol(i) for i in range(self.psi4mol.natom())]
        self.orientation_options.generate_orientations(symbols)
        dct = self.orientation_options.to_indices()
        xyzs = []
        for a, b, c in dct["reorientations"]:
            xyzs.append(utils.orient_rigid(a, b, c, self.coordinates))
        for a, b, c in dct["rotations"]:
            xyzs.append(utils.rotate_rigid(a, b, c, self.coordinates))
        for translation in dct["translations"]:
            xyzs.append(self.coordinates + translation)
        
        for coordinates in xyzs:
            self._add_orientation(coordinates)


    @base.datafile(filename="optimized_geometry.xyz")
    def compute_opt_mol(self):
        tmpdir = self.directory
        infile = f"{self.name}_opt.in"
        outfile = self.qm_options.write_opt_file(self.psi4mol,
                                                 destination_dir=tmpdir,
                                                 filename=infile)


        cmd = f"cd {tmpdir}; psi4 -i {infile} -o {outfile}; cd -"
        # maybe it's already run?
        if not self.io_options.force and os.path.isfile(outfile):
            return utils.log2xyz(outfile)
        subprocess.run(cmd, shell=True)
        return utils.log2xyz(outfile)

    def compute_optimized_geometry(self, executor=None):
        if not self.optimized and self.optimize_geometry:
            try:
                future = executor.submit(self.compute_opt_mol)
                future.add_done_callback(lambda x: self.update_geometry_from_xyz(x.result()))
            except AttributeError:
                xyz = self.compute_opt_mol()
                self.update_geometry_from_xyz(xyz)


    def update_geometry_from_xyz(self, xyz):
        mol = psi4.core.Molecule.from_string(xyz, dtype="xyz")
        self.psi4mol.set_geometry(mol.geometry())
        self.optimized = True
        self.add_orientations()

    def compute_unweighted_a_matrix(self):
        A = np.zeros((self.n_atoms, self.n_atoms))
        for mol in self.orientations:
            A += mol.get_esp_mat_a()
        return A

    def compute_unweighted_b_matrix(self, executor=None):
        B = np.zeros(self.n_atoms)
        get_esp_mat_bs = [x.get_esp_mat_b for x in self.orientations]
        try:
            futures = list(map(executor.submit, get_esp_mat_bs))
            for future in as_completed(futures):
                B += future.result()
        except AttributeError:
            for mol in self.orientations:
                B += mol.get_esp_mat_b()
        return B

    def get_unweighted_b_matrix(self, executor=None):
        if self._unweighted_b_matrix is None:
            self._unweighted_b_matrix = self.compute_unweighted_b_matrix()
        return self._unweighted_b_matrix

    def get_unweighted_a_matrix(self):
        if self._unweighted_a_matrix is None:
            self._unweighted_a_matrix = self.compute_unweighted_a_matrix()
        return self._unweighted_a_matrix

    def get_weighted_a_matrix(self, executor=None):
        return self.get_unweighted_a_matrix() * (self.weight ** 2)

    def get_weighted_b_matrix(self, executor=None):
        return self.get_unweighted_b_matrix(executor=executor) * (self.weight ** 2)

    @property
    def vdw_points(self):
        if self._vdw_points is None:
            self._vdw_points = self.compute_vdw_points()
        return self._vdw_points

    def compute_vdw_points(self):
        el = [self.psi4mol.symbol(i) for i in range(self.psi4mol.natom())]
        return utils.gen_connolly_shells(el,
                                         vdw_radii=self.esp_options.vdw_radii,
                                         use_radii=self.esp_options.use_radii,
                                         scale_factors=self.esp_options.vdw_scale_factors,
                                         density=self.esp_options.vdw_point_density)
