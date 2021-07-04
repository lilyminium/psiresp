import pytest
import psiresp

from numpy.testing import assert_almost_equal

# from .datafiles import DMSO, DMSO_O1, DMSO_O2, DMSO_O3, DMSO_O4


class TestNoOrient:

    esp_1 = [-0.43877469, 0.14814998, 0.17996033, 0.18716814, 0.35743529,
             -0.5085439, -0.46067469, 0.19091725, 0.15500465, 0.18935764]

    resp_1 = [-0.31436216, 0.11376836, 0.14389443, 0.15583112, 0.30951582,
              -0.50568553, -0.33670393, 0.15982115, 0.12029174, 0.153629]

    esp_2 = [-0.39199538, 0.15716631, 0.15716631, 0.15716631, 0.35743529,
             -0.5085439, -0.43701446, 0.16953984, 0.16953984, 0.16953984]

    resp_2 = [-0.25158642, 0.11778735, 0.11778735, 0.11778735, 0.30951582,
              -0.50568553, -0.29298059, 0.12912489, 0.12912489, 0.12912489]

    def test_esp_1(self, dmso_psi4mol):
        # options = psiresp.ConformerOptions(optimize_geometry=False)
        resp = psiresp.Resp(psi4mol=dmso_psi4mol,
                            restrained=True, stage_2=True,
                            conformer_options=dict(optimize_geometry=False))
        resp.run()
        assert_almost_equal(resp.stage_1_charges.unrestrained_charges,
                            self.esp_1, decimal=5)
        assert_almost_equal(resp.stage_1_charges.restrained_charges,
                            self.resp_1, decimal=5)
        assert_almost_equal(resp.stage_2_charges.unrestrained_charges,
                            self.esp_2, decimal=5)
        assert_almost_equal(resp.stage_2_charges.restrained_charges,
                            self.resp_2, decimal=5)
