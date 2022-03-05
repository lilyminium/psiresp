from ast import keyword
import pytest

# from numpy.testing import assert_allclose

# from psiresp.qm import QMEnergyOptions

# pytest.importorskip("psi4")

from psiresp.qm import QMEnergyOptions, QMGeometryOptimizationOptions


@pytest.mark.parametrize("qm_options", [QMEnergyOptions, QMGeometryOptimizationOptions])
@pytest.mark.parametrize("keywords", [{}, {"maxiter": 300}])
def test_pass_in_keywords(qm_options, keywords):
    options = qm_options(keywords=keywords).generate_keywords()
    for k, v in keywords.items():
        assert options[k] == v

# @pytest.mark.skip("hangs in CI")
# class TestQMEnergyOptions:

#     @pytest.fixture()
#     def cheap_options(self):
#         return QMEnergyOptions(basis="sto-3g", method="b3lyp")

#     def test_add_compute(self, cheap_options, dmso_qcmol, empty_client):
#         molhash = dmso_qcmol.get_hash()
#         records = empty_client.query_molecules(molecule_hash=[molhash])
#         assert len(records) == 0
#         response = cheap_options.add_compute(empty_client, qcmols=[dmso_qcmol])
#         records = empty_client.query_results(id=response.submitted)
#         assert len(records) == 1
#         record = records[0]

#         assert record.status == "INCOMPLETE"
#         assert record.get_molecule() == dmso_qcmol

#     def test_add_compute_and_wait(self, cheap_options, dmso_qcmol, empty_client):
#         # molhash = dmso_qcmol.get_hash()
#         # records = empty_client.query_molecules(molecule_hash=[molhash])
#         # assert len(records) == 0
#         records = cheap_options.add_compute_and_wait(empty_client, qcmols=[dmso_qcmol])
#         assert len(records) == 1

#         record = records[0]
#         assert record.status == "COMPLETE"
#         assert record.get_molecule() == dmso_qcmol
#         assert record.wavefunction is not None
#         assert_allclose(record.properties.return_energy, -546.68015604292)
