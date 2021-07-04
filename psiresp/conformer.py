
import logging
import io
from typing import Optional, List, Any

import numpy as np
import MDAnalysis as mda
from pydantic import PrivateAttr

from . import mixins
from .utils import orientation as orutils
from .utils import psi4utils
from .utils.io import datafile
from .orientation import Orientation

logger = logging.getLogger(__name__)


class Conformer(mixins.MoleculeMixin, mixins.ConformerOptions):
    """Class to manage one conformer
    """
    resp: Any  # TODO: resp typing

    _orientations: List[Orientation] = PrivateAttr(default_factory=list)
    _finalized: bool = PrivateAttr(default=False)
    _unweighted_a_matrix: Optional[np.ndarray] = PrivateAttr(default=None)
    _unweighted_b_matrix: Optional[np.ndarray] = PrivateAttr(default=None)

    @property
    def orientations(self):
        return self._orientations

    @orientations.setter
    def orientations(self, value):
        self._orientations = value

    def _empty_init(self):
        self._unweighted_a_matrix = None
        self._unweighted_b_matrix = None

    @property
    def default_path(self):
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

    @datafile(filename="optimized_geometry.xyz")
    def compute_optimized_geometry(self):
        if not self.optimize_geometry:
            return psi4utils.psi4mol_to_xyz_string(self.psi4mol)
        with self.directory() as tmpdir:
            infile, outfile = self.resp.resp.write_opt_file(self.psi4mol)
            self.resp.resp.try_run_qm(infile, outfile=outfile, cwd=tmpdir)
            xyz = psi4utils.opt_logfile_to_xyz_string(outfile)
        return xyz

    def add_orientation(self,
                        coordinates_or_psi4mol: psi4utils.CoordinateInputs,
                        name: Optional[str] = None,
                        **kwargs) -> Orientation:
        """Create Orientation from Psi4 molecule or coordinates and add

        Parameters
        ----------
        coordinates_or_psi4mol: numpy.ndarray of coordinates or psi4.core.Molecule
            An array of coordinates or a Psi4 Molecule. If this is a molecule,
            the molecule is copied before creating the Conformer.
        name: str (optional)
            Name of the conformer. If not provided, one will be generated
            from the name template in the conformer_generator
        **kwargs:
            Arguments used to construct the Orientation.
            If not provided, the default specification given in
            :attr:`psiresp.conformer.Conformer.orientation_options`
            will be used.

        Returns
        -------
        orientation: Orientation
        """
        if name is None:
            counter = len(self.orientations) + 1
            name = self.orientation_name_template.format(conformer=self,
                                                         counter=counter)
        mol = psi4utils.psi4mol_with_coordinates(self.psi4mol,
                                                 coordinates_or_psi4mol,
                                                 name=name)
        default_kwargs = self.orientation_options.to_kwargs(**kwargs)
        orientation = Orientation(conformer=self, psi4mol=mol, name=name,
                                  **default_kwargs)
        self.orientations.append(orientation)
        return orientation

    def generate_orientations(self):
        """Generate Orientations for this conformer"""
        all_coordinates = self.generate_orientation_coordinates()
        if len(all_coordinates) > len(self.orientations):
            self._orientations = []
            for coordinates in all_coordinates:
                self.add_orientation(coordinates)

            if not self.orientations:
                self.add_orientation(self.psi4mol)

    def finalize_geometry(self):
        """Finalize geometry of psi4mol

        If :attr:`psiresp.conformer.Conformer.optimize_geometry` is ``True``,
        this will optimize the geometry using Psi4. If not, this will continue
        using the current geometry.
        If :attr:`psiresp.conformer.Conformer.save_output` is ``True``, the
        final geometry will get written to an xyz file to bypass this check
        next time.
        """
        xyz = self.compute_optimized_geometry()
        mol = psi4utils.psi4mol_from_xyz_string(xyz)
        self.psi4mol.set_geometry(mol.geometry())
        self._finalized = True
        self._empty_init()
        self.generate_orientations()

    def compute_unweighted_a_matrix(self) -> np.ndarray:
        """Average the inverse squared distance matrices
        from each orientation to generate the A matrix
        for solving Ax = B.

        Returns
        -------
        numpy.ndarray
            The shape of this array is (n_atoms, n_atoms)
        """
        A = np.zeros((self.n_atoms, self.n_atoms))
        for mol in self.orientations:
            A += mol.get_esp_mat_a()
        return A  # / self.n_orientations

    def compute_unweighted_b_matrix(self) -> np.ndarray:
        """Average the ESP by distance from each orientation
        to generate the B vector for solving Ax = B

        Returns
        -------
        numpy.ndarray
            The shape of this vector is (n_atoms,)
        """
        B = np.zeros(self.n_atoms)
        get_esp_mat_bs = [x.get_esp_mat_b for x in self.orientations]
        for mol in self.orientations:
            B += mol.get_esp_mat_b()
        return B  # / self.n_orientations

    @property
    def transformations(self):
        return [self.reorientations, self.rotations, self.translations]

    @property
    def n_specified_transformations(self):
        return sum(map(len, self.transformations))

    @property
    def n_transformations(self):
        return sum([self.n_rotations, self.n_translations, self.n_reorientations])

    def generate_transformations(self):
        """Generate atom combinations and coordinates for transformations.

        This is used to create the number of transformations specified
        by n_rotations, n_reorientations and n_translations if these
        transformations are not already given.
        """
        for kw in ("reorientations", "rotations"):
            target = getattr(self, f"n_{kw}")
            container = getattr(self, kw)
            n = max(target - len(container), 0)
            combinations = orutils.generate_atom_combinations(self.symbols)
            while len(container) < target:
                container.append(next(combinations))

        n_trans = self.n_translations - len(self.translations)
        if n_trans > 0:
            new_translations = (np.random.rand(n_trans, 3) - 0.5) * 10
            self.translations.extend(new_translations)

    @datafile(filename="orientation_coordinates.npy")
    def generate_orientation_coordinates(self):
        coordinates = self.coordinates
        self.generate_transformations()

        transformed = []
        if self.keep_original_conformer_geometry:
            transformed.append(coordinates)
        for reorient in self.reorientations:
            indices = orutils.id_to_indices(reorient)
            new = orutils.orient_rigid(*indices, coordinates)
            transformed.append(new)

        for rotate in self.rotations:
            indices = orutils.id_to_indices(rotate)
            new = orutils.rotate_rigid(*indices, coordinates)
            transformed.append(new)

        for translate in self.translations:
            transformed.append(coordinates + translate)

        if not transformed:
            transformed.append(coordinates)

        return np.array(transformed)
