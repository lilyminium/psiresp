import psiresp
import pytest
import os
import numpy as np

from numpy.testing import (assert_almost_equal, assert_equal,
                           assert_allclose)
from .utils import (mol_from_file, esp_from_gamess_file,
                    coordinates_from_xyz)


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

    @pytest.fixture(scope='function')
    def coordinates(self):
        return coordinates_from_xyz('{}.xyz'.format(self.orientname))

    def test_correct_coordinates(self, opt_orientation, coordinates):
        assert_almost_equal(opt_orientation.coordinates, coordinates)

    def test_get_grid(self, opt_orientation, esp):
        ref = esp[:, 1:]
        grid = opt_orientation.get_grid()
        assert_almost_equal(grid, ref, decimal=4)

    def test_get_esp_gas(self, opt_orientation, esp, tmpdir):
        ref = esp[:, 0]
        with tmpdir.as_cwd():
            # opt_orientation.get_esp_matrices()
            epot = opt_orientation.esp
        assert_almost_equal(epot, ref, decimal=4)


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
        return psiresp.Orientation(geometry, load_files=False)

    def test_init_orientation(self, orientation):
        assert orientation.name == 'default'
        assert orientation.n_atoms == 10
        assert_equal(orientation.indices, np.arange(10).astype(int))
        assert_equal(orientation.symbols, list('CHHHSOCHHH'))
        assert orientation.grid is None
        assert orientation.esp is None
        assert orientation.r_inv is None


@pytest.mark.fast
class TestOrientationDMSO2(TestOrientationDMSO1):
    orientname = 'dmso_opt_c1_o2'
