import pytest
import itertools
import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal


import psiresp

from .base import charges_from_red_file
from .datafiles import (ETHANOL_RESP2_C1, ETHANOL_RESP2_C2,
                        TEST_RESP2_DATA, TEST_MULTIRESP2_DATA,
                        ETHANOL_RESP2_GAS_STAGE1_MATRICES,
                        ETHANOL_RESP2_GAS_C1_STAGE1_MATRICES,
                        ETHANOL_RESP2_GAS_C1_O1_GRID_ESP,
                        ETHANOL_RESP2_GAS_C1_O1_GRID,
                        AMM_NME_OPT_RESPA1_CHARGES,
                        
                        )

ETOH_SOLV_CHARGES = np.array([-0.2416,  0.3544, -0.6898,  0.0649,  0.0649,
                              0.0649, -0.0111, -0.0111,  0.4045])
ETOH_GAS_CHARGES = np.array([-0.2300,  0.3063, -0.5658,  0.0621,  0.0621,
                             0.0621, -0.0153, -0.0153,  0.3339])
ETOH_REF_CHARGES = np.array([-0.2358,  0.33035, -0.6278,  0.0635,
                             0.0635,  0.0635, -0.0132, -0.0132,  0.3692])


@pytest.fixture()
def etoh_resp2():
    resp2 = psiresp.Resp2.from_molfile(ETHANOL_RESP2_C1, ETHANOL_RESP2_C2,
                                       fix_geometry=True,
                                       name="resp2_ethanol",
                                       load_input=True,
                                       directory_path=TEST_RESP2_DATA,
                                       delta=0.5)
    resp2.generate_orientations()
    return resp2


class TestResp2:

    def test_resp2_construction(self, etoh_resp2):
        assert len(etoh_resp2.gas.conformers) == 2
        assert len(etoh_resp2.solvated.conformers) == 2
        assert etoh_resp2.name == "resp2_ethanol"
        assert str(etoh_resp2.path) == TEST_RESP2_DATA

        gas_path = f"{TEST_RESP2_DATA}/resp2_ethanol_gas"
        gas_phase = etoh_resp2.gas
        assert str(gas_phase.path) == gas_path
        assert str(gas_phase.conformers[0].path) == f"{gas_path}/resp2_ethanol_gas_c001"

        assert all(len(conf.orientations) == 1 for conf in etoh_resp2.conformers)
        orientation = etoh_resp2.gas.conformers[0].orientations[0]
        path = (f"{TEST_RESP2_DATA}/resp2_ethanol_gas/"
                "resp2_ethanol_gas_c001/"
                "resp2_ethanol_gas_c001_o001")
        assert str(orientation.path) == path

        # assert gas_phase.grid_rmin == 1.3
        assert gas_phase.solvent == "water"

        assert np.allclose(etoh_resp2.gas.conformer_coordinates[0, 0, 0], 1.059)

    def test_grid_generation(self, etoh_resp2):
        orientation = etoh_resp2.gas.conformers[0].orientations[0]
        expected_esp = np.loadtxt(ETHANOL_RESP2_GAS_C1_O1_GRID_ESP)
        assert_allclose(orientation.esp, expected_esp)
        expected_grid = np.loadtxt(ETHANOL_RESP2_GAS_C1_O1_GRID)
        assert_allclose(orientation.grid, expected_grid)

        assert np.allclose(orientation.coordinates[0, 0], 1.059)
        assert np.allclose(orientation.grid[0, 0], 1.7023625732724663)
        assert np.allclose(orientation.r_inv[0, 0], 0.22234337)

        gen_grid = etoh_resp2.grid_options.generate_vdw_grid(orientation.symbols,
                                                             orientation.coordinates)
        assert_allclose(gen_grid, expected_grid)


    def test_resp2_gas_conformer(self, etoh_resp2):
        a = etoh_resp2.gas.conformers[0].unweighted_a_matrix
        b = etoh_resp2.gas.conformers[0].unweighted_b_matrix

        expected_ab = np.loadtxt(ETHANOL_RESP2_GAS_C1_STAGE1_MATRICES)
        A = expected_ab[:-1]
        B = expected_ab[-1]

        assert_allclose(a, A)
        assert_allclose(b, B)

    def test_resp2_gas(self, etoh_resp2):
        a = etoh_resp2.gas.get_a_matrix()
        b = etoh_resp2.gas.get_b_matrix()

        expected_ab = np.loadtxt(ETHANOL_RESP2_GAS_STAGE1_MATRICES)
        A = expected_ab[:-1]
        B = expected_ab[-1]

        assert_allclose(a, A)
        assert_allclose(b, B)

    def test_resp2_run(self, etoh_resp2):
        etoh_resp2.run()

        assert etoh_resp2.gas._stage_2_charges is not etoh_resp2.solvated._stage_2_charges

        # not quite sure when rounding happens with original RESP2 implementation
        assert_allclose(etoh_resp2.solvated.charges, ETOH_SOLV_CHARGES, atol=5e-03)
        assert_allclose(etoh_resp2.gas.charges, ETOH_GAS_CHARGES, atol=5e-03)
        assert_allclose(etoh_resp2.charges, ETOH_REF_CHARGES, atol=5e-03)


class TestMultiResp2Ethanol:

    @pytest.fixture()
    def etoh_multiresp2(self, etoh_resp2):
        multiresp = psiresp.MultiResp2([etoh_resp2], delta=0.5, directory_path=TEST_RESP2_DATA,
                                       name="")
        multiresp.generate_orientations()
        return multiresp

    def test_multiresp_construction(self, etoh_multiresp2):
        assert len(etoh_multiresp2.resps) == 1
        assert len(etoh_multiresp2.gas.resps) == 1
        assert len(etoh_multiresp2.gas.resps[0].conformers) == 2
        assert len(etoh_multiresp2.gas.resps[0].conformers[0].orientations) == 1

        orientation = etoh_multiresp2.gas.resps[0].conformers[0].orientations[0]
        path = (f"{TEST_RESP2_DATA}/resp2_ethanol_gas/"
                "resp2_ethanol_gas_c001/"
                "resp2_ethanol_gas_c001_o001")
        assert str(orientation.path) == path

        expected_esp = np.loadtxt(ETHANOL_RESP2_GAS_C1_O1_GRID_ESP)
        assert_allclose(orientation.esp, expected_esp)
        expected_grid = np.loadtxt(ETHANOL_RESP2_GAS_C1_O1_GRID)
        assert_allclose(orientation.grid, expected_grid)

        assert np.allclose(orientation.coordinates[0, 0], 1.059)
        assert np.allclose(orientation.grid[0, 0], 1.7023625732724663)
        assert np.allclose(orientation.r_inv[0, 0], 0.22234337)

    def test_multiresp_run(self, etoh_multiresp2):
        etoh_multiresp2.run()
        assert_allclose(etoh_multiresp2.solvated.charges, ETOH_SOLV_CHARGES, atol=5e-03)
        assert_allclose(etoh_multiresp2.gas.charges, ETOH_GAS_CHARGES, atol=5e-03)
        assert_allclose(etoh_multiresp2.charges, ETOH_REF_CHARGES, atol=5e-03)


def test_multiple_multiresp2(nme2ala2_opt_resp,
                              methylammonium_resp):

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

    multiresp = psiresp.MultiResp2(resps=[methylammonium_resp, nme2ala2_opt_resp],
                                   charge_constraint_options=overall, delta=0.0,
                                   load_input=True, save_output=True,
                                   directory_path=TEST_MULTIRESP2_DATA)
    assert multiresp.qm_options.solvent == "water"
    assert multiresp.gas.qm_options.solvent == "water"
    assert multiresp.gas.resps[0].resp.qm_options.solvent == "water"
    assert multiresp.gas.resps[0].resp.grid_options is multiresp.gas.grid_options

    multiresp.generate_orientations()
    assert len(list(multiresp.conformers)) == 6
    assert len(list(multiresp.orientations)) == 20

    for orientation in multiresp.orientations:
        assert "gas" in orientation.name or "solvated" in orientation.name

    charges = multiresp.run()
    assert_almost_equal(charges[[0, 1, 2, 3, 8, 9, 10, 11, 12, 13, 14, 15]].sum(), 0)
    assert_almost_equal(charges[[27, 28, 29, 30, 31, 32]].sum(), 0)
    assert_almost_equal(charges[25], 0.6163)
    assert_almost_equal(charges[26], -0.5722)
    assert_almost_equal(charges[18], charges[22])

    assert not np.allclose(multiresp.gas.charges, multiresp.solvated.charges)

    # can't really compare these to the 6-31g* ones
    # check constraints worked instead
    assert_allclose(charges[17], charges[21])
    for a, b in itertools.combinations([5, 6, 7], 2):
        assert_allclose(charges[a], charges[b])
    
    for a, b in itertools.combinations([18, 19, 20, 22, 23, 24], 2):
        assert_allclose(charges[a], charges[b])