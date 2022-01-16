import pytest
import qcelemental as qcel
from numpy.testing import assert_allclose

from psiresp.tests.datafiles import DMSO, DMSO_ESP

from psiresp.orientation import Orientation
from psiresp.qm import QMEnergyOptions
from psiresp.tests.utils import load_gamess_esp

pytest.importorskip("psi4")


def test_compute_esp_regression(fractal_client):
    # setup
    qcmol = qcel.models.Molecule.from_file(DMSO)
    ref = load_gamess_esp(DMSO_ESP)
    ref_grid = ref[:, 1:]
    ref_esp = ref[:, 0]
    orientation = Orientation(qcmol=qcmol, grid=ref_grid)

    # checks
    assert orientation.energy is None
    assert orientation.esp is None

    qm_options = QMEnergyOptions()
    result_id = qm_options.add_compute(fractal_client, qcmols=[orientation.qcmol]).ids
    record = qm_options.wait_for_results(fractal_client, response_ids=result_id)[0]

    orientation.compute_esp_from_record(record)
    assert_allclose(orientation.esp, ref_esp, atol=1e-07)
    assert orientation.energy < 0


def test_compute_esp(methylammonium_empty, fractal_client, job_grids, job_esps):
    orientation = methylammonium_empty.conformers[0].orientations[0]
    mol_ids = fractal_client.add_molecules([orientation.qcmol])
    record = fractal_client.query_results(id=mol_ids)[0]

    fname = orientation.qcmol.get_hash()
    orientation.compute_grid()
    assert_allclose(orientation.grid, job_grids[fname], atol=5e-9)

    orientation.compute_esp_from_record(record)
    assert_allclose(orientation.esp, job_esps[fname], atol=5e-9)
