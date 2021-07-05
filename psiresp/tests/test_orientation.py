

import pytest
import numpy as np
from numpy.testing import assert_almost_equal

from psiresp.tests.datafiles import (DMSO, DMSO_ESP, DMSO_RINV,
                                     DMSO_O1, DMSO_O1_ESP, DMSO_O1_RINV,
                                     DMSO_O2, DMSO_O2_ESP, DMSO_O2_RINV,
                                     )
from psiresp.tests.base import (coordinates_from_xyzfile,
                                psi4mol_from_xyzfile,
                                orientation_from_psi4mol,
                                esp_from_gamess_file
                                )


class BaseTestOrientation:
    xyzfile = None
    espfile = None
    rinv_file = None

    @pytest.fixture()
    def gamess_esp(self):
        return esp_from_gamess_file(self.espfile)

    @pytest.fixture()
    def grid(self, gamess_esp):
        return gamess_esp[:, 1:]

    @pytest.fixture()
    def esp(self, gamess_esp):
        return gamess_esp[:, 0]

    @pytest.fixture()
    def r_inv(self):
        return np.loadtxt(self.rinv_file)

    @pytest.fixture()
    def orientation(self, esp):
        psi4mol = psi4mol_from_xyzfile(self.xyzfile)
        orientation = orientation_from_psi4mol(psi4mol)
        orientation._esp = esp
        return orientation

    def test_coordinates(self, orientation):
        coordinates = coordinates_from_xyzfile(self.xyzfile)
        assert_almost_equal(orientation.coordinates, coordinates)

    def test_grid(self, orientation, grid):
        assert_almost_equal(orientation.grid, grid, decimal=5)

    @pytest.mark.slow
    def test_esp(self, grid, esp):
        psi4mol = psi4mol_from_xyzfile(self.xyzfile)
        orientation = orientation_from_psi4mol(psi4mol)
        orientation._grid = grid
        assert_almost_equal(orientation.esp, esp, decimal=4)

    def test_rinv(self, orientation, r_inv):
        assert_almost_equal(orientation.r_inv, r_inv, decimal=5)


class TestOrientationDMSO(BaseTestOrientation):
    xyzfile = DMSO
    espfile = DMSO_ESP
    rinv_file = DMSO_RINV


class TestOrientationDMSO_O1(BaseTestOrientation):
    xyzfile = DMSO_O1
    espfile = DMSO_O1_ESP
    rinv_file = DMSO_O1_RINV


class TestOrientationDMSO_O2(BaseTestOrientation):
    xyzfile = DMSO_O2
    espfile = DMSO_O2_ESP
    rinv_file = DMSO_O2_RINV
