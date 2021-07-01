from dataclasses import dataclass, field
from typing import Tuple, Dict

import numpy as np


from . import vdwradii, base


class GridMixin(base.Model):
    """Options for setting up the grid for ESP computation

    Parameters
    ----------
    grid_rmin: float
        minimum radius
    grid_rmax: float
        maximum radius
    use_radii: str
        Name of the radius set to use
    vdw_radii: dict of {str: float}
        Dictionary of VDW radii to override the radii in the
        `use_radii` set
    vdw_scale_factors: iterable of floats
        Scale factors for the radii
    vdw_point_density: float
        Point density
    """

    grid_rmin: float = 0
    grid_rmax: float = -1
    use_radii: str = "msk"
    vdw_radii: Dict[str, float] = {}
    vdw_scale_factors: Tuple[float] = (1.4, 1.6, 1.8, 2.0)
    vdw_point_density: float = 1.0

    @property
    def effective_rmax(self):
        return self.grid_rmax if self.grid_rmax >= 0 else np.inf

    def __post_init__(self):
        if self.effective_rmax < self.grid_rmin:
            raise ValueError(f"grid_rmax (given: {self.grid_rmax}) must be equal to "
                             f"or greater than grid_rmin (given: {self.grid_rmin})")

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
        unique_radii, inverse = np.unique(radii, return_inverse=True)
        surface_areas = (unique_radii ** 2) * np.pi * 4
        n_points = (surface_areas * self.vdw_point_density).astype(int)
        points = [self.generate_unit_sphere(n) * radius
                  for n, radius in zip(n_points, unique_radii)]
        nan_points = np.empty((len(radii), max(n_points), 3)) * np.nan
        for i in inverse:
            i_points = points[i]
            nan_points[i][:i_points[0], :i_points[1]] = i_points
        return nan_points * radii

    def get_shell_within_bounds(self,
                                radii: np.ndarray,
                                spheres: np.ndarray,
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

        spheres = spheres.transpose((1, 0, 2))  # n_points, n_atoms, 3
        shell = spheres + coordinates
        distances = np.linalg.norm(shell - coordinates, axis=2)
        within_bounds = (distances >= inner_bound) & (distances <= outer_bound)
        inside = np.all(within_bounds, axis=1)
        return shell[inside].reshape((-1, 3))

    def generate_vdw_grid(self,
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
        base_spheres = self.generate_connolly_spheres(symbol_radii)

        points = []

        for factor in self.scale_factors:
            radii = symbol_radii * factor
            spheres = base_spheres * factor
            shell = self.get_shell_within_bounds(radii, spheres, coordinates)
            points.append(shell)

        return np.concatenate(points)
