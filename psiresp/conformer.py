import logging
from typing import Optional

import numpy as np

from . import psi4utils, mixins, options
from .orientation import Orientation

logger = logging.getLogger(__name__)


class Conformer(options.ConformerOptions, mixins.MoleculeMixin):
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

    @mixins.io.datafile(filename="optimized_geometry.xyz")
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
            name = self.orientation_options.format_name(conformer=self,
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
        self.orientations = []
        coords = self.orientation_generator.get_transformed_coordinates(self.symbols,
                                                                        self.coordinates)
        for coordinates in coords:
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
        mol = psi4.core.Molecule.from_string(xyz, dtype="xyz")
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
        return A / self.n_orientations

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
        return B / self.n_orientations
