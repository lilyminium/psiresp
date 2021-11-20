import pytest

import qcfractal.interface as ptl
from psiresp.testing import TemporaryPostgres
from qcfractal import FractalSnowflake

from psiresp.tests.datafiles import POSTGRES_SERVER_BACKUP


@pytest.fixture(scope="session")
def fractal_client():
    storage = TemporaryPostgres(database_name="test_psiresp")
    storage.psql.restore_database(POSTGRES_SERVER_BACKUP)
    postgres_server = storage.psql
    with FractalSnowflake(
        max_workers=1,
        storage_project_name="test_psiresp",
        storage_uri=postgres_server.database_uri(),
        reset_database=False,
        start_server=True,
    ) as server:
        yield ptl.FractalClient(server)
    storage.stop()


@pytest.fixture(scope="function")
def empty_client():
    with FractalSnowflake(max_workers=1, start_server=True) as server:
        yield ptl.FractalClient(server)
