

import pytest
import numpy as np
import qcelemental as qcel
from numpy.testing import assert_allclose

from psiresp.tests.datafiles import (DMSO, DMSO_ESP, DMSO_RINV,
                                     DMSO_O1, DMSO_O1_ESP, DMSO_O1_RINV,
                                     DMSO_O2, DMSO_O2_ESP, DMSO_O2_RINV,
                                     )
# from psiresp.tests.base import (coordinates_from_xyzfile,
#                                 psi4mol_from_xyzfile,
#                                 orientation_from_psi4mol,
#                                 esp_from_gamess_file
#                                 )

from psiresp.tests.utils import load_gamess_esp

@pytest.mark.parametrize("coord_file, esp_file", [
    (DMSO, DMSO_ESP),
    (DMSO_O1, DMSO_O1_ESP),
    (DMSO_O2, DMSO_O2_ESP),
])
def test_compute_esp(coord_file, esp_file, fractal_client):
    # setup
    qcmol = qcel.models.Molecule.from_file(coord_file)
    ref = load_gamess_esp(esp_file)
    ref_grid = ref[:, 1:]
    ref_esp = ref[:, 0]
    orientation = Orientation(qcmol=qcmol, grid=ref_grid)

    # checks
    assert orientation.energy is None
    assert orientation.esp is None

    record = fractal_client.query_molecules(molecule_hash=qcmol.get_hash())
    orientation.compute_esp(record)
    assert_allclose(orientation.esp, ref_esp)
    assert orientation.energy < 0
