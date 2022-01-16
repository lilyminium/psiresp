from psiresp.tests.datafiles import (DMSO, DMSO_ESP,
                                     DMSO_O1, DMSO_O1_ESP,
                                     DMSO_O2, DMSO_O2_ESP,
                                     )
import pytest
from numpy.testing import assert_allclose, assert_equal


psi4utils = pytest.importorskip("psiresp.psi4utils")


def test_psi4mol_from_qcmol(dmso_qcmol):
    psi4mol = psi4utils.psi4mol_from_qcmol(dmso_qcmol)
    symbols = [psi4mol.symbol(i) for i in range(psi4mol.natom())]
    assert psi4mol.natom() == 10
    assert symbols == list("CHHHSOCHHH")


def test_get_connectivity(dmso_qcmol):
    connectivity = psi4utils.get_connectivity(dmso_qcmol)
    assert connectivity.shape == (9, 3)


# def test_get_sp3_ch_indices(dmso_qcmol):
#     groups = psi4utils.get_sp3_ch_indices(dmso_qcmol)
#     reference = {
#         0: [1, 2, 3],
#         6: [7, 8, 9],
#     }
#     assert groups.keys() == reference.keys()
#     for k, v in groups.items():
#         assert_equal(v, reference[k])


@pytest.mark.xfail(reason="fix qcrecord lookup and server")
@pytest.mark.parametrize("qcrecord, reference_grid, reference_esp", [
    (DMSO, DMSO_ESP, DMSO_ESP),
    (DMSO_O1, DMSO_O1_ESP, DMSO_O1_ESP),
    (DMSO_O2, DMSO_O2_ESP, DMSO_O2_ESP),
], indirect=True)
def test_compute_esp(qcrecord, reference_grid, reference_esp):
    esp = psi4utils.compute_esp(qcrecord, reference_grid)
    assert_allclose(esp, reference_esp, atol=1e-10)
