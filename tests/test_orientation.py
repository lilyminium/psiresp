import psiresp
import pytest
import os
import numpy as np

from numpy.testing import (assert_almost_equal, assert_equal,
                           assert_allclose)
from .utils import (mol_from_file, esp_from_gamess_file,
                    coordinates_from_xyz, datafile)


class BaseTestOrientation:

    molname = None
    orientname = None

    @pytest.fixture(scope='class')
    def opt_mol(self):
        return mol_from_file('{}.xyz'.format(self.orientname))

    @pytest.fixture(scope='function')
    def opt_orientation(self, opt_mol):
        return psiresp.Orientation(opt_mol, load_files=False)

    @pytest.fixture(scope='function')
    def esp(self):
        return esp_from_gamess_file('{}.esp'.format(self.orientname))
    
    @pytest.fixture(scope="function")
    def r_inv(self):
        filename = datafile(f"{self.orientname}_r_inv.dat")
        return np.loadtxt(filename)

    @pytest.fixture(scope='function')
    def coordinates(self):
        return coordinates_from_xyz('{}.xyz'.format(self.orientname))

    def test_correct_coordinates(self, opt_orientation, coordinates):
        assert_almost_equal(opt_orientation.coordinates, coordinates)

    def test_get_grid(self, opt_orientation, esp, tmpdir):
        ref = esp[:, 1:]
        with tmpdir.as_cwd():
            assert_almost_equal(opt_orientation.grid, ref, decimal=4)

    def test_get_esp(self, opt_orientation, esp, tmpdir):
        ref = esp[:, 0]
        with tmpdir.as_cwd():
            assert_almost_equal(opt_orientation.esp, ref, decimal=4)

    def test_get_rinv(self, opt_orientation, r_inv, tmpdir):
        with tmpdir.as_cwd():
            calc_r_inv = opt_orientation.r_inv
        assert_almost_equal(calc_r_inv, r_inv)
        

@pytest.mark.fast
class TestOrientationDMSO0(BaseTestOrientation):
    orientname = 'dmso_opt_c1'


@pytest.mark.fast
class TestOrientationDMSO1(BaseTestOrientation):
    molname = 'dmso_c1'
    orientname = 'dmso_opt_c1_o1'

    @pytest.fixture(scope='class')
    def geometry(self):
        return mol_from_file('{}.xyz'.format(self.molname))

    @pytest.fixture(scope='function')
    def orientation(self, geometry):
        return psiresp.Orientation(geometry)

    def test_init_orientation(self, orientation):
        assert orientation.name == 'default'
        assert orientation.n_atoms == 10
        assert_equal(orientation.indices, np.arange(10).astype(int))
        assert_equal(orientation.symbols, list('CHHHSOCHHH'))


@pytest.mark.fast
class TestOrientationDMSO2(TestOrientationDMSO1):
    orientname = 'dmso_opt_c1_o2'
