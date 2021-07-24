import pytest
import numpy as np
from numpy.testing import assert_allclose

import psiresp

from .datafiles import (ETHANOL_RESP2_C1, ETHANOL_RESP2_C2,
                        TEST_RESP2_DATA,
                        ETHANOL_RESP2_GAS_STAGE1_MATRICES,
                        ETHANOL_RESP2_GAS_C1_STAGE1_MATRICES,
                        ETHANOL_RESP2_GAS_C1_O1_GRID_ESP,
                        ETHANOL_RESP2_GAS_C1_O1_GRID,
                        )


@pytest.fixture()
def etoh_resp2():
    resp2 = psiresp.Resp2.from_molfile(ETHANOL_RESP2_C1, ETHANOL_RESP2_C2,
                                       name="resp2_ethanol",
                                       load_input=True,
                                       directory_path=TEST_RESP2_DATA,
                                       delta=0.5)
    resp2.generate_orientations()
    return resp2


def test_resp2_construction(etoh_resp2):
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

    assert gas_phase.grid_rmin == 1.3
    assert gas_phase.solvent == "water"

    expected_esp = np.loadtxt(ETHANOL_RESP2_GAS_C1_O1_GRID_ESP)
    assert_allclose(orientation.esp, expected_esp)
    expected_grid = np.loadtxt(ETHANOL_RESP2_GAS_C1_O1_GRID)
    assert_allclose(orientation.grid, expected_grid)


def test_resp2_gas_conformer(etoh_resp2):
    a = etoh_resp2.gas.conformers[0].unweighted_a_matrix
    b = etoh_resp2.gas.conformers[0].unweighted_b_matrix

    expected_ab = np.loadtxt(ETHANOL_RESP2_GAS_C1_STAGE1_MATRICES)
    A = expected_ab[:-1]
    B = expected_ab[-1]

    assert_allclose(a, A)
    assert_allclose(b, B)


def test_resp2_gas(etoh_resp2):
    a = etoh_resp2.gas.get_a_matrix()
    b = etoh_resp2.gas.get_b_matrix()

    expected_ab = np.loadtxt(ETHANOL_RESP2_GAS_STAGE1_MATRICES)
    A = expected_ab[:-1]
    B = expected_ab[-1]

    assert_allclose(a, A)
    assert_allclose(b, B)


# def test_resp2_run(etoh_resp2):
#     etoh_resp2.run()
#     SOLV = np.array([-0.2416,  0.3544, -0.6898,  0.0649,  0.0649,
#                      0.0649, -0.0111, -0.0111,  0.4045])
#     GAS = np.array([-0.2300,  0.3063, -0.5658,  0.0621,  0.0621,
#                     0.0621, -0.0153, -0.0153,  0.3339])
#     REF = np.array([-0.2358,  0.33035, -0.6278,  0.0635,
#                     0.0635,  0.0635, -0.0132, -0.0132,  0.3692])

#     assert etoh_resp2.gas._stage_2_charges is not etoh_resp2.solvated._stage_2_charges
#     assert_allclose(etoh_resp2.solvated.charges, SOLV)
#     assert_allclose(etoh_resp2.gas.charges, GAS)
#     assert_allclose(etoh_resp2.charges, REF)
