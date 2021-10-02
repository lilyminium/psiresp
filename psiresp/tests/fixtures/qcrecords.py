import pytest

import qcelemental as qcel


@pytest.fixture
def qcrecord(request, fractal_client):
    qcmol = qcel.models.Molecule.from_file(request.param).get_hash()
    records = fractal_client.query_results()
    molecules = [rec.get_molecule().get_hash() for rec in records]
    for i, molhash in enumerate(molecules):
        if molhash == qcmol:
            return records[i]
    raise ValueError("QCRecord not found")
