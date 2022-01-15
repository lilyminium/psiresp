import glob
import pytest


import numpy as np

from psiresp.tests.datafiles import POSTGRES_SERVER_BACKUP, ESP_PATH, GRID_PATH

pytest_plugins = [
    "psiresp.tests.fixtures.qcmols",
    "psiresp.tests.fixtures.qcrecords",
    "psiresp.tests.fixtures.molecules",
    "psiresp.tests.fixtures.options",
]


# @pytest.fixture(scope="session")
# def postgres_server():
#     storage = TemporaryPostgres(database_name="test_psiresp")
#     storage.psql.restore_database(POSTGRES_SERVER_BACKUP)
#     return storage.psql
#     storage.stop()


@pytest.fixture(scope="session")
def fractal_client():
    pytest.importorskip("qcfractal.postgres_harness")
    from qcfractal import FractalSnowflake, interface as ptl
    from psiresp.testing import TemporaryPostgres

    storage = TemporaryPostgres(database_name="test_psiresp")
    storage.psql.restore_database(POSTGRES_SERVER_BACKUP)
    postgres_server = storage.psql
    server = FractalSnowflake(
        max_workers=1,
        storage_project_name="test_psiresp",
        storage_uri=postgres_server.database_uri(),
        reset_database=False,
        start_server=False)
    return ptl.FractalClient(server)


@pytest.fixture(scope="function")
def empty_client():
    pytest.importorskip("qcfractal.snowflake")
    import qcfractal.interface as ptl
    from qcfractal import FractalSnowflake

    server = FractalSnowflake()
    return ptl.FractalClient(server)


# @pytest.fixture(scope="session")
# def fractal_client(postgres_server):
#     with FractalSnowflake(
#         max_workers=1,
#         storage_project_name="test_psiresp",
#         storage_uri=postgres_server.database_uri(),
#         reset_database=False,
#         start_server=False,
#     ) as server:
#         yield ptl.FractalClient(server)


# @pytest.fixture(scope="function")
# def empty_client():
#     with FractalSnowflake() as server:
#         yield ptl.FractalClient(server)


@pytest.fixture
def reference_esp(request):
    return np.loadtxt(request.param, comments='!')[:, 0]


@pytest.fixture
def reference_grid(request):
    return np.load(request.param)


@pytest.fixture
def red_charges(request):
    with open(request.param, 'r') as f:
        content = f.read()

    mols = [x.split('\n')[1:] for x in content.split('MOLECULE') if x]
    charges = [np.array([float(x.split()[4]) for x in y if x]) for y in mols]
    # if len(charges) == 1:
    #     charges = charges[0]
    return charges


@pytest.fixture
def job_esps():
    mol_esps = {}
    for fname in glob.glob(ESP_PATH):
        qchash = fname.split("/")[-1].split("_")[0]
        mol_esps[qchash] = np.loadtxt(fname)
    return mol_esps


@pytest.fixture
def job_grids():
    mol_esps = {}
    for fname in glob.glob(GRID_PATH):
        qchash = fname.split("/")[-1].split("_")[0]
        mol_esps[qchash] = np.loadtxt(fname)
    return mol_esps
