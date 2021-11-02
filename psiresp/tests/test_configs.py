import pytest
import numpy as np
from numpy.testing import assert_allclose


import psiresp

from psiresp.tests.datafiles import (AMM_NME_OPT_ESPA1_CHARGES,
                                     AMM_NME_OPT_RESPA2_CHARGES,
                                     AMM_NME_OPT_RESPA1_CHARGES,
                                     )


@pytest.mark.parametrize("config_class, red_charges", [
    (psiresp.configs.EspA1, AMM_NME_OPT_ESPA1_CHARGES),
    (psiresp.configs.RespA2, AMM_NME_OPT_RESPA2_CHARGES),
    (psiresp.configs.RespA1, AMM_NME_OPT_RESPA1_CHARGES),
], indirect=['red_charges'])
def test_config_multiresp(nme2ala2, methylammonium,
                          methylammonium_nme2ala2_charge_constraints,
                          config_class, red_charges,
                          job_esps, job_grids):

    job = config_class(molecules=[methylammonium, nme2ala2],
                       charge_constraints=methylammonium_nme2ala2_charge_constraints)
    assert isinstance(job, config_class)

    for orient in job.iter_orientations():
        fname = orient.qcmol.get_hash()
        orient.esp = job_esps[fname]
        orient.grid = job_grids[fname]

    job.compute_charges()
    charges = np.concatenate(job.charges)
    # print(charges)
    # print(methylammonium_nme2ala2_charge_constraints)
    print(hash(methylammonium), hash(job.molecules[0]))
    print(hash(nme2ala2), hash(job.molecules[1]))
    print("IN")
    print(methylammonium)
    print("\n==OUT==")
    print(job.molecules[0])
    # print(job.stage_1_charges)

    assert_allclose(charges[[0, 1, 2, 3, 8, 9, 10, 11, 12, 13, 14, 15]].sum(), 0, atol=1e-7)
    assert_allclose(charges[[27, 28, 29, 30, 31, 32]].sum(), 0, atol=1e-7)
    assert_allclose(charges[25], 0.6163)
    assert_allclose(charges[26], -0.5722)
    assert_allclose(charges[18], charges[22])
    for calculated, reference in zip(job.charges[::-1], red_charges[::-1]):
        assert_allclose(calculated, reference, atol=1e-3)
