import pytest

from numpy.testing import assert_allclose

from psiresp.job import Job
from psiresp.resp import RespOptions


class TestSingleResp:
    def test_unrestrained(self, dmso, fractal_client):
        options = RespOptions(stage_2=True, restrained_fit=False)
        job = Job(molecules=[dmso],
                  resp_options=options)
        job.run(client=fractal_client)

        esp_1 = [-0.43877469, 0.14814998, 0.17996033, 0.18716814, 0.35743529,
                 -0.5085439, -0.46067469, 0.19091725, 0.15500465, 0.18935764]
        esp_2 = [-0.39199538, 0.15716631, 0.15716631, 0.15716631, 0.35743529,
                 -0.5085439, -0.43701446, 0.16953984, 0.16953984, 0.16953984]
        assert_allclose(job.stage_1_charges.unrestrained_charges, esp_1)
        assert_allclose(job.stage_2_charges.unrestrained_charges, esp_2)

    def test_restrained(self, dmso, fractal_client):
        options = RespOptions(stage_2=True, restrained_fit=True)
        job = Job(molecules=[dmso],
                  resp_options=options)
        job.run(client=fractal_client)

        resp_1 = [-0.31436216, 0.11376836, 0.14389443, 0.15583112, 0.30951582,
                  -0.50568553, -0.33670393, 0.15982115, 0.12029174, 0.153629]
        resp_2 = [-0.25158642, 0.11778735, 0.11778735, 0.11778735, 0.30951582,
                  -0.50568553, -0.29298059, 0.12912489, 0.12912489, 0.12912489]
        assert_allclose(job.stage_1_charges.restrained_charges, resp_1)
        assert_allclose(job.stage_2_charges.restrained_charges, resp_2)
