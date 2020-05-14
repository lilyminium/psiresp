import pytest
import os
import numpy as np

from psiresp import utils
from numpy.testing import assert_almost_equal
from .utils import coordinates_from_xyz, datafile

@pytest.mark.fast
@pytest.mark.parametrize('molname,onum,orient', [
    ('dmso_opt_c1', 1, (1, 5, 6)),
    ('dmso_opt_c1', 2, (6, 5, 1)),
])
def test_orient_rigid(molname, onum, orient):
    original = coordinates_from_xyz('{}.xyz'.format(molname))
    i, j, k = np.asarray(orient)-1
    oriented = utils.orient_rigid(i, j, k, original)
    ref = coordinates_from_xyz('{}_o{}.xyz'.format(molname, onum))
    assert_almost_equal(oriented, ref, decimal=5)

@pytest.mark.fast
@pytest.mark.parametrize('molname,onum,rotate', [
    ('dmso_opt_c1', 3, (1, 5, 6)),
    ('dmso_opt_c1', 4, (6, 5, 1)),
])
def test_rotate_rigid(molname, onum, rotate):
    original = coordinates_from_xyz('{}_qmra.xyz'.format(molname))
    i, j, k = np.asarray(rotate)-1
    oriented = utils.rotate_rigid(i, j, k, original)
    ref = coordinates_from_xyz('{}_o{}.xyz'.format(molname, onum))
    assert_almost_equal(oriented, ref, decimal=5)

@pytest.mark.fast
@pytest.mark.parametrize('factor,radii_name,scaled', [
    (1.4, 'msk', [1.68, 2.1, 1.96]),
    (1.6, 'msk', [1.92, 2.4, 2.24]),
    (1.8, 'msk', [2.16, 2.7, 2.52]),
    (2.0, 'msk', [2.4, 3., 2.8]),
    (1.4, 'bondi', [1.68, 2.38, 2.128]),
    (1.6, 'bondi', [1.92, 2.72, 2.432]),
    (1.8, 'bondi', [2.16, 3.06, 2.736]),
    (2.0, 'bondi', [2.4, 3.4, 3.04])
])
def test_scale_radii(factor, radii_name, scaled):
    radii = utils.scale_radii({'H', 'C', 'O'}, factor,
                              use_radii=radii_name)
    ref = {k: v for k, v in zip('HCO', scaled)}
    assert len(ref) == len(radii)
    for k, refv in ref.items():
        assert_almost_equal(radii[k], refv)

@pytest.mark.fast
@pytest.mark.parametrize('n', [3, 10, 29, 44, 48, 64])
def test_gen_unit_sphere(n):
    points = utils.gen_unit_sphere(n)
    fn = datafile('surface_n{}.dat'.format(n))
    ref = np.loadtxt(fn, comments='!')
    assert_almost_equal(points, ref, decimal=5)
    assert len(points) <= n

@pytest.mark.fast
@pytest.mark.parametrize('scaled,density,n_points', [
    ([1.68, 2.1, 1.96], 0.0, [0, 0, 0]),
    ([1.68, 2.1, 1.96], 1.0, [35, 55, 48]),
    ([1.68, 2.1, 1.96], 2.0, [70, 110, 96]),
    ([2.4, 3., 2.8], 3.4, [246, 384, 334]),
])
def test_gen_connolly_spheres(scaled, density, n_points):
    radii = {k: v for k, v in zip('HCO', scaled)}
    points, rad = utils.gen_connolly_spheres('CHOHCC', radii,
                                             density=density)
    assert len(rad) == 6
    assert len(points) == 6

    point_arr = np.asarray(n_points)[[1, 0, 2, 0, 1, 1]]
    for pt, n in zip(points, point_arr):
        assert len(pt) <= n and len(pt) >= 0.85*n

@pytest.mark.fast
@pytest.mark.parametrize('vdw_radii,scale_factors,density,radii,n_points', [
    ({}, (1.4, 2.0), 1.0,
     [[1.68, 2.1, 1.96], [2.4, 3., 2.8]],
     [[35, 55, 48], [72, 113, 98]]),
    ({}, (1.4, 2.0), 2.0,
     [[1.68, 2.1, 1.96], [2.4, 3., 2.8]],
     [[70, 110, 96], [144, 226, 197]]),
    ({'C': 1}, (2.0,), 2.0, [[2.4, 2, 2.8]], [[144, 100, 197]]),
])
def test_gen_connolly_shells(vdw_radii, scale_factors, density, radii,
                             n_points):
    points = utils.gen_connolly_shells('CHOHCC', vdw_radii=vdw_radii,
                                       scale_factors=scale_factors,
                                       density=density)
    assert len(points) == len(scale_factors)
    assert all(len(spheres[0]) == 6 for spheres in points)
    for (spheres, rads), ns, scaled in zip(points, n_points, radii):
        point_arr = np.asarray(ns)[[1, 0, 2, 0, 1, 1]]
        rad_arr = np.asarray(scaled)[[1, 0, 2, 0, 1, 1]]
        assert_almost_equal(rads, rad_arr)
        for pt, n in zip(spheres, point_arr):
            assert len(pt) <= n and len(pt) >= 0.85*n

# TODO: test_gen_vdw_surface
