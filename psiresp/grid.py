"""
VdW grid options --- :mod:`psiresp.grid`
===============================================

This module contains options and code for generating van der Waals'
Connolly shells around the molecule. Users should not typically
interact with the methods on this class themselves, but instead simply
set the options and call :meth:`psiresp.job.Job.run`.

.. autoclass:: psiresp.grid.GridOptions
    :members:
"""

from typing import Dict, List, Any, Optional, Union, TYPE_CHECKING

import numpy as np
from scipy.spatial import distance as spdist
from pydantic import Field
import qcelemental as qcel

from . import vdwradii, base

if TYPE_CHECKING:  # pragma: no cover
    import qcelemental
    import qcfractal


class GridOptions(base.Model):
    """Options for setting up the grid for ESP computation"""

    grid_rmin: float = Field(
        default=0,
        description="Minimum radius around atom for constructing shells"
    )
    grid_rmax: float = Field(
        default=-1,
        description=("Maximum radius around atom for constructing shells. "
                     "If set to -1, no points are clipped.")
    )
    use_radii: vdwradii.VdwRadiiSet = Field(
        default="msk",
        description="Name of the radius set to use",
    )
    vdw_radii: Dict[str, float] = Field(
        default_factory=dict,
        description=("Dictionary of specific VDW radii to override the radii "
                     "the `use_radii` set"),
    )
    vdw_scale_factors: List[float] = Field(
        default=[1.4, 1.6, 1.8, 2.0],
        description="Scale factors used to generate grid shells",
    )
    vdw_point_density: float = Field(
        default=1.0,
        description="Point density in the generated grid shells"
    )

    @property
    def effective_rmax(self):
        return self.grid_rmax if self.grid_rmax >= 0 else np.inf

    @property
    def all_vdw_radii(self):
        new = dict(**vdwradii.options[self.use_radii.lower()])
        new.update(self.vdw_radii)
        return new

    def get_vdwradii_for_elements(self, symbols: List[str]) -> np.ndarray:
        """Return an array of VdW radii for the specified elements"""
        given_radii = self.all_vdw_radii
        symbols = [x.capitalize() for x in symbols]
        missing = set([el for el in symbols if el not in given_radii])
        if missing:
            err = (f"{', '.join(missing)} are not supported elements. "
                   "Specify the radius in the ``vdw_radii`` dictionary.")
            raise KeyError(err)

        return np.array([given_radii[el] for el in symbols])

    @staticmethod
    def generate_unit_sphere(n_points):
        """
        Get coordinates of n points on a unit sphere.

        Adapted from GAMESS.

        Parameters
        ----------
        n_points: int
            maximum number of points

        Returns
        -------
        coordinates: np.ndarray
            cartesian coordinates of points
        """
        INCREMENT = 1e-10
        n_latitude_points = int((np.pi * n_points) ** 0.5)
        n_longitude_points = int(n_latitude_points / 2)

        rows = np.arange(n_longitude_points + 1)
        vertical_angles = rows * np.pi / n_longitude_points
        z = np.cos(vertical_angles)
        xy = np.sin(vertical_angles)
        n_points_per_row = (xy * n_latitude_points + INCREMENT).astype(int)
        n_points_per_row[n_points_per_row < 1] = 1
        circum_fraction = [np.arange(n_in_row) / n_in_row
                           for n_in_row in n_points_per_row]
        row_points = np.concatenate(circum_fraction) * 2 * np.pi
        n_specific_points = sum(n_points_per_row)

        points = np.empty((n_specific_points, 3))
        points[:, -1] = np.repeat(z, n_points_per_row)
        all_xy = np.repeat(xy, n_points_per_row)
        points[:, 0] = np.cos(row_points) * all_xy
        points[:, 1] = np.sin(row_points) * all_xy

        return points[:n_points]

    def generate_connolly_spheres(self, radii):
        """
        Compute Connolly spheres of specified radii and density around each atom.

        Parameters
        ----------
        radii: 1D numpy.ndarray of floats
            scaled radii of elements

        Returns
        -------
        points: list of numpy.ndarray of shape (N, 3)
            cartesian coordinates of points
        """
        radii = np.asarray(radii)
        unique_radii, inverse = np.unique(radii, return_inverse=True)
        surface_areas = (unique_radii ** 2) * np.pi * 4
        n_points = (surface_areas * self.vdw_point_density).astype(int)
        nan_points = np.full((len(unique_radii), max(n_points), 3), np.nan)
        for i, n in enumerate(n_points):
            shell = self.generate_unit_sphere(n)
            nan_points[i][:len(shell)] = shell
        nan_points *= unique_radii.reshape((-1, 1, 1))
        # nan_points = nan_points[:, ~np.all(np.isnan(nan_points), axis=(0, 2))]
        return nan_points[inverse]

    def get_shell_within_bounds(self,
                                radii: np.ndarray,
                                coordinates: np.ndarray,
                                ) -> np.ndarray:
        """
        Filter shell points to lie between `inner_bound` and `outer_bound`

        Parameters
        ----------
        radii: numpy.ndarray
            This has shape (N,) where N is the number of atoms
        spheres: numpy.ndarray
            This has shape (N, M, 3)
        coordinates: numpy.ndarray
            This has shape (N, 3)

        Returns
        -------
        numpy.ndarray
            with shape (L, 3)
        """

        inner_bound = radii * self.grid_rmin
        inner_bound = np.where(inner_bound < radii, radii, inner_bound)
        outer_bound = radii * self.effective_rmax

        spheres = self.generate_connolly_spheres(radii)
        shell = spheres + coordinates.reshape((-1, 1, 3))  # n_atoms, n_points, 3
        shell_points = np.concatenate(shell)
        # shell_points = shell_points[~np.isnan(shell_points)]

        # we want to ignore self-to-self false negatives
        # so we mask all distances calculated from an atom's sphere to the atom
        # x, y form the mask
        n_atoms, n_points, _ = spheres.shape
        atom_indices = np.arange(n_atoms)
        y = np.repeat(atom_indices, n_points)
        x = np.arange(len(shell_points))

        distances = spdist.cdist(shell_points, coordinates)  # n_points, n_atoms
        distances[np.isnan(distances)] = -1
        within_bounds = (distances >= inner_bound) & (distances <= outer_bound)
        within_bounds[(x, y)] = True
        inside = np.all(within_bounds, axis=1)
        return shell_points[inside]

    def _generate_vdw_grid(self,
                           symbols: List[str],
                           coordinates: np.ndarray,
                           ) -> np.ndarray:
        """Generate VdW surface points

        Parameters
        ----------
        symbols: list of str
            Atom elements
        coordinates: numpy.ndarray
            This has shape (N, 3)

        Returns
        -------
        numpy.ndarray
            This has shape (M, 3)
        """
        symbol_radii = self.get_vdwradii_for_elements(symbols)

        points = []
        for factor in self.vdw_scale_factors:
            radii = symbol_radii * factor
            points.extend(self.get_shell_within_bounds(radii, coordinates))
        return np.array(points)

    def generate_grid(self,
                      molecule: Union[str, int,
                                      Dict[str, Any],
                                      "qcelemental.models.Molecule",
                                      "qcfractal.interface.models.records.ResultRecord",
                                      ],
                      client: Optional["qcfractal.interface.client.FractalClient"] = None,
                      ) -> np.ndarray:
        """Generate VdW surface points
        """
        if isinstance(molecule, dict):
            molecule = qcel.models.Molecule.from_data(molecule, validate=True)
        elif isinstance(molecule, (str, int)) and client is not None:
            molecule = client.query_molecules(id=[str(molecule)])[0]
        elif not isinstance(molecule, qcel.models.Molecule):
            try:
                molecule = molecule.get_molecule()
            except AttributeError:
                try:
                    molecule = molecule.to_qcschema()
                except AttributeError:
                    raise ValueError("`molecule` must be a "
                                     "QCElemental molecule, "
                                     "QCFractal ResultRecord, "
                                     "or OpenFF Toolkit Molecule.")

        bohr2angstrom = qcel.constants.conversion_factor("bohr", "angstrom")
        coordinates = molecule.geometry * bohr2angstrom
        grid = self._generate_vdw_grid(molecule.symbols, coordinates)
        return grid
