import logging

import numpy as np

from . import base, psi4utils, mixins

from .orientation import Orientation

logger = logging.getLogger(__name__)

class ConformerOptions(mixins.IOMixin):
    optimize_geometry: bool = False
    weight: float = 1
    orientation_options: OrientationOptions = OrientationOptions()
    orientation_generator: OrientationGenerator = OrientationGenerator()

class Conformer(ConformerOptions, mixins.MoleculeMixin):
    resp: "Resp"

    def __post_init__(self):
        super().__post_init__()
        self.orientations = []
        self._finalized = False
        self._empty_init()

    def _empty_init(self):
        self._unweighted_a_matrix = None
        self._unweighted_b_matrix = None

    @property
    def path(self):
        return self.resp.path / self.name

    @property
    def unweighted_a_matrix(self):
        if self._unweighted_a_matrix is None:
            self._unweighted_a_matrix = self.compute_unweighted_a_matrix()
        return self._unweighted_a_matrix
    
    @property
    def unweighted_b_matrix(self):
        if self._unweighted_b_matrix is None:
            self._unweighted_b_matrix = self.compute_unweighted_b_matrix()
        return self._unweighted_b_matrix

    @property
    def weighted_a_matrix(self):
        return self.unweighted_a_matrix * (self.weight ** 2)
    
    @property
    def weighted_b_matrix(self):
        return self.unweighted_b_matrix * (self.weight ** 2)

    @property
    def n_orientations(self):
        return len(self.orientations)

    @base.datafile(filename="optimized_geometry.xyz")
    def compute_optimized_geometry(self):
        if not self.optimize_geometry:
            return psi4utils.psi4mol_to_xyz_string(self.psi4mol)
        with self.directory() as tmpdir:
            infile, outfile = self.qm_options.write_opt_file(self.psi4mol)
            self.try_run_qm(infile, outfile=outfile, cwd=tmpdir)
            xyz = psi4utils.psi4logfile_to_xyz_string(outfile)
        return xyz

    def add_orientation(self, coordinates_or_psi4mol, name=None, **kwargs):
        if name is None:
            counter = len(self.orientations) + 1
            name = self.orientation_options.format_name(conformer=self,
                                                        counter=counter)
        mol = psi4utils.psi4mol_with_coordinates(self.psi4mol,
                                                 coordinates_or_psi4mol,
                                                 name=name)
        orientation = Orientation(conformer=self, psi4mol=mol, name=name)
        self.orientations.append(orientation)

    def generate_orientations(self):
        """Generate Orientations for this conformer"""
        self.orientations = []
        coords = self.orientation_generator.get_transformed_coordinates(self.symbols,
                                                                        self.coordinates)
        for coordinates in coords:
            self.add_orientation(coordinates)

        if not self.orientations:
            self.add_orientation(self.psi4mol)

    def finalize_geometry(self):
        xyz = self.compute_optimized_geometry()
        mol = psi4.core.Molecule.from_string(xyz, dtype="xyz")
        self.psi4mol.set_geometry(mol.geometry())
        self._finalized = True
        self._empty_init()
        self.generate_orientations()

    def compute_unweighted_a_matrix(self):
        A = np.zeros((self.n_atoms, self.n_atoms))
        for mol in self.orientations:
            A += mol.get_esp_mat_a()
        return A / self.n_orientations

    def compute_unweighted_b_matrix(self):
        B = np.zeros(self.n_atoms)
        get_esp_mat_bs = [x.get_esp_mat_b for x in self.orientations]
        for mol in self.orientations:
            B += mol.get_esp_mat_b()
        return B / self.n_orientations

