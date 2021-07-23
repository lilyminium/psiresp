import pytest
from numpy.testing import assert_almost_equal

import numpy as np
from psiresp import mixins

from ..datafiles import UNIT_SPHERE_3, UNIT_SPHERE_64, DMSO_SHELL_D1, DMSO_SHELL_D2


@pytest.mark.parametrize("n_points, ref_file", [
    (3, UNIT_SPHERE_3),
    (64, UNIT_SPHERE_64)
])
def test_generate_unit_sphere(n_points, ref_file):
    points = mixins.GridOptions.generate_unit_sphere(n_points)
    reference = np.loadtxt(ref_file, comments="!")
    assert_almost_equal(points, reference, decimal=5)


@pytest.mark.parametrize("radii, density, n_points", [
    ([1.68, 2.1, 1.96], 0.0, [0, 0, 0]),
    ([1.68, 2.1, 1.96], 1.0, [35, 55, 48]),
    ([1.68, 2.1, 1.96], 2.0, [70, 110, 96]),
    ([2.4, 3., 2.8], 3.4, [246, 384, 334]),
])
def test_generate_connolly_spheres(radii, density, n_points):
    grid = mixins.GridOptions(vdw_point_density=density)
    points = grid.generate_connolly_spheres(radii)
    mask = np.all(np.isnan(points), axis=(0, 2))
    assert points.shape == ((3, n_points[1], 3))
    for sphere, n in zip(points, n_points):
        col = sphere[:, 0]
        assert len(col[~np.isnan(col)]) <= n
