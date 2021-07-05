import pytest
import psiresp

import numpy as np
from numpy.testing import assert_almost_equal

from .base import charges_from_red_file
from .datafiles import (AMM_NME_OPT_RESPA2_CHARGES,
                        AMM_NME_OPT_RESPA1_CHARGES,
                        AMM_NME_OPT_ESPA1_CHARGES)


@pytest.mark.parametrize("stage_2, hyp_a1, charge_file", [
    (False, 0.0, AMM_NME_OPT_ESPA1_CHARGES),
    (False, 0.01, AMM_NME_OPT_RESPA2_CHARGES),
    (True, 0.0005, AMM_NME_OPT_RESPA1_CHARGES),

])
def test_separate_charge_constraints(nme2ala2_opt_resp,
                                     methylammonium_resp,
                                     stage_2, hyp_a1, charge_file):
    options1 = dict(charge_constraints=[(0, [20, 21, 22, 23, 24, 25]),
                                        (0.6163, [18]),
                                        (-0.5722, [19])],
                    charge_equivalences=[(10, 14), (11, 12, 13, 15, 16, 17)])
    options2 = dict(charge_equivalences=[(6, 7, 8)])
    overall = dict(charge_constraints=[[0, [(1, 1), (1, 2), (1, 3), (1, 4),
                                            (2, 1), (2, 2), (2, 3), (2, 4),
                                            (2, 5), (2, 6), (2, 7), (2, 8)]]],
                   symmetric_methyls=False)
    nme2ala2_opt_resp.charge_constraint_options = options1
    methylammonium_resp.charge_constraint_options = options2
    multiresp = psiresp.MultiResp(resps=[methylammonium_resp, nme2ala2_opt_resp],
                                  charge_constraint_options=overall)
    multiresp.stage_2 = stage_2
    multiresp.hyp_a1 = hyp_a1
    charges = multiresp.run()
    assert_almost_equal(charges[[0, 1, 2, 3, 8, 9, 10, 11, 12, 13, 14, 15]].sum(), 0)
    assert_almost_equal(charges[[27, 28, 29, 30, 31, 32]].sum(), 0)
    assert_almost_equal(charges[25], 0.6163)
    assert_almost_equal(charges[26], -0.5722)
    assert_almost_equal(charges[18], charges[22])

    reference = np.concatenate(charges_from_red_file(charge_file))
    assert_almost_equal(charges, reference, decimal=3)


@pytest.mark.parametrize("stage_2, hyp_a1, charge_file", [
    (False, 0.0, AMM_NME_OPT_ESPA1_CHARGES),
    (False, 0.01, AMM_NME_OPT_RESPA2_CHARGES),
    (True, 0.0005, AMM_NME_OPT_RESPA1_CHARGES),

])
def test_combined_constraints(nme2ala2_opt_resp,
                              methylammonium_resp,
                              stage_2, hyp_a1, charge_file):

    overall = dict(charge_constraints=[(0, [(1, 1), (1, 2), (1, 3), (1, 4),
                                            (2, 1), (2, 2), (2, 3), (2, 4),
                                            (2, 5), (2, 6), (2, 7), (2, 8)]),

                                       (0, [(2, 20), (2, 21), (2, 22),
                                            (2, 23), (2, 24), (2, 25)]),
                                       (0.6163, [(2, 18)]),
                                       (-0.5722, [(2, 19)]),
                                       ],
                   charge_equivalences=[[(2, 10), (2, 14)],
                                        [(2, 11), (2, 12), (2, 13),
                                         (2, 15), (2, 16), (2, 17)],
                                        [(1, 6), (1, 7), (1, 8)],
                                        ],
                   symmetric_methyls=False)

    multiresp = psiresp.MultiResp(resps=[methylammonium_resp, nme2ala2_opt_resp],
                                  charge_constraint_options=overall)
    multiresp.stage_2 = stage_2
    multiresp.hyp_a1 = hyp_a1
    charges = multiresp.run()
    assert_almost_equal(charges[[0, 1, 2, 3, 8, 9, 10, 11, 12, 13, 14, 15]].sum(), 0)
    assert_almost_equal(charges[[27, 28, 29, 30, 31, 32]].sum(), 0)
    assert_almost_equal(charges[25], 0.6163)
    assert_almost_equal(charges[26], -0.5722)
    assert_almost_equal(charges[18], charges[22])

    reference = np.concatenate(charges_from_red_file(charge_file))
    assert_almost_equal(charges, reference, decimal=3)
