import pytest

from numpy.testing import assert_allclose

from psiresp.qm import QMEnergyOptions

pytest.importorskip("psi4")


class TestQMEnergyOptions:

    @pytest.fixture()
    def cheap_options(self):
        return QMEnergyOptions(basis="sto-3g", method="b3lyp")

    def test_add_compute(self, cheap_options, dmso_qcmol, empty_client):
        molhash = dmso_qcmol.get_hash()
        records = empty_client.query_molecules(molecule_hash=[molhash])
        assert len(records) == 0
        response = cheap_options.add_compute(empty_client, qcmols=[dmso_qcmol])
        records = empty_client.query_results(id=response.submitted)
        assert len(records) == 1
        record = records[0]

        assert record.status == "INCOMPLETE"
        assert record.get_molecule() == dmso_qcmol

    @pytest.mark.slow
    def test_add_compute_and_wait(self, cheap_options, dmso_qcmol, empty_client):
        molhash = dmso_qcmol.get_hash()
        records = empty_client.query_molecules(molecule_hash=[molhash])
        assert len(records) == 0
        records = cheap_options.add_compute_and_wait(empty_client, qcmols=[dmso_qcmol])
        assert len(records) == 1

        record = records[0]
        assert record.status == "COMPLETE"
        assert record.get_molecule() == dmso_qcmol
        assert record.wavefunction is not None
        assert_allclose(record.properties.return_energy, -546.68015604292)
