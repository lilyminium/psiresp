import pytest
from numpy.testing import assert_allclose

import qcelemental as qcel

import numpy as np
from psiresp.grid import GridOptions

from psiresp.tests.datafiles import (UNIT_SPHERE_3, UNIT_SPHERE_64,
                                     DMSO, DMSO_GRID,
                                     DMSO_O1, DMSO_O1_GRID,
                                     DMSO_O2, DMSO_O2_GRID,
                                     )


@pytest.fixture()
def default_grid_options():
    return GridOptions()


@pytest.mark.parametrize("n_points, ref_file", [
    (3, UNIT_SPHERE_3),
    (64, UNIT_SPHERE_64)
])
def test_generate_unit_sphere(n_points, ref_file):
    points = GridOptions.generate_unit_sphere(n_points)
    reference = np.loadtxt(ref_file, comments="!")
    assert_allclose(points, reference, atol=1e-10)


@pytest.mark.parametrize("radii, density, n_points", [
    ([1.68, 2.1, 1.96], 0.0, [0, 0, 0]),
    ([1.68, 2.1, 1.96], 1.0, [35, 55, 48]),
    ([1.68, 2.1, 1.96], 2.0, [70, 110, 96]),
    ([2.4, 3., 2.8], 3.4, [246, 384, 334]),
])
def test_generate_connolly_spheres(radii, density, n_points):
    grid = GridOptions(vdw_point_density=density)
    points = grid.generate_connolly_spheres(radii)
    mask = np.all(np.isnan(points), axis=(0, 2))
    assert points.shape == ((3, n_points[1], 3))
    for sphere, n in zip(points, n_points):
        col = sphere[:, 0]
        assert len(col[~np.isnan(col)]) <= n


@pytest.mark.xfail(reason="Incorrect reference grids calculated in bohr")
@pytest.mark.parametrize("qcmol, reference_grid", [
    (DMSO, DMSO_GRID),
    (DMSO_O1, DMSO_O1_GRID),
    (DMSO_O2, DMSO_O2_GRID),
], indirect=True)
def test_generate_vdw_grid(qcmol, reference_grid, default_grid_options):
    grid = default_grid_options.generate_grid(qcmol)
    assert_allclose(grid, reference_grid)
