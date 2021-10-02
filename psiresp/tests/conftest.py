import shutil
from pkg_resources import resource_filename

import pytest

import numpy as np
import qcfractal.interface as ptl
import qcelemental as qcel
from qcfractal import FractalSnowflake, FractalSnowflakeHandler
from psiresp.testing import TemporaryPostgres
from psiresp.molecule import Molecule

from psiresp.tests.datafiles import POSTGRES_SERVER_BACKUP, DMSO
from psiresp.tests.utils import load_gamess_esp

pytest_plugins = [
    "psiresp.tests.fixtures.qcmols",
    "psiresp.tests.fixtures.qcrecords",
    "psiresp.tests.fixtures.molecules",
]


@pytest.fixture(scope="session")
def postgres_server():
    storage = TemporaryPostgres(database_name="test_psiresp")
    # storage.psql.restore_database(POSTGRES_SERVER_BACKUP)
    yield storage.psql
    # storage.psql.backup_database(POSTGRES_SERVER_BACKUP)
    storage.stop()


@pytest.fixture(scope="session")
def fractal_server(postgres_server):
    with FractalSnowflake(
        max_workers=1,
        storage_project_name="test_psiresp",
        storage_uri=postgres_server.database_uri(),
        reset_database=False,
        start_server=True,
    ) as server:
        yield server


@pytest.fixture(scope="session")
def fractal_client(fractal_server):
    yield ptl.FractalClient(fractal_server)


@pytest.fixture(scope="function")
def empty_client():
    return FractalSnowflakeHandler().client()


@pytest.fixture
def reference_esp(request):
    return np.loadtxt(request.param, comments='!')[:, 0]


@pytest.fixture
def reference_grid(request):
    return np.load(request.param)
