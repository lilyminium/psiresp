import pytest

import qcfractal.interface as ptl
from psiresp.testing import TemporaryPostgres
from qcfractal import FractalSnowflake

from psiresp.tests.datafiles import POSTGRES_SERVER_BACKUP


@pytest.fixture(scope="session")
def postgres_server():
    storage = TemporaryPostgres(database_name="test_psiresp")
    storage.psql.restore_database(POSTGRES_SERVER_BACKUP)
    yield storage.psql
    storage.stop()


@pytest.fixture(scope="session")
def fractal_client(postgres_server):
    with FractalSnowflake(
        max_workers=1,
        storage_project_name="test_psiresp",
        storage_uri=postgres_server.database_uri(),
        reset_database=False,
        start_server=False,
    ) as server:
        yield ptl.FractalClient(server)


@pytest.fixture(scope="function")
def empty_client():
    with FractalSnowflake() as server:
        yield ptl.FractalClient(server)
