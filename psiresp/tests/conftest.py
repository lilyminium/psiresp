import shutil
from pkg_resources import resource_filename

import pytest

import qcfractal.interface as ptl
from qcfractal.postgres_harness import TemporaryPostgres

from .datafiles import TMP_POSTGRES_SERVER


@pytest.fixture(scope="session")
def tmp_postgres_server():
    if shutil.which("psql") is None:
        pytest.skip("Postgres is not installed on this server and no active postgres could be found.")

    storage = TemporaryPostgres(database_name="test_psiresp",
                                tmpdir=TMP_POSTGRES_SERVER)
    yield storage.psql
    storage.stop()


@pytest.fixture(scope="session")
def tmp_server(tmp_postgres_server):
    with FractalSnowflake(
        max_workers=2,
        storage_project_name="test_psiresp",
        storage_uri=tmp_postgres_server.database_uri(),
        reset_database=False,
        start_server=False,
    ) as server:
        yield server


@pytest.fixture(scope="session")
def tmp_client(tmp_server):
    yield ptl.FractalClient(tmp_server)
