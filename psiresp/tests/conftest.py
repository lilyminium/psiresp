import shutil
import glob
from pkg_resources import resource_filename

import pytest


import numpy as np
import qcfractal.interface as ptl
import qcelemental as qcel
from qcfractal import FractalSnowflake, FractalSnowflakeHandler
from psiresp.testing import TemporaryPostgres
from psiresp.molecule import Molecule

from psiresp.tests.datafiles import POSTGRES_SERVER_BACKUP, DMSO, ESP_PATH, GRID_PATH
from psiresp.tests.utils import load_gamess_esp

pytest_plugins = [
    "psiresp.tests.fixtures.qcmols",
    "psiresp.tests.fixtures.qcrecords",
    "psiresp.tests.fixtures.molecules",
    "psiresp.tests.fixtures.options",
]


@pytest.fixture(scope="session")
def postgres_server():
    storage = TemporaryPostgres(database_name="test_psiresp")
    storage.psql.restore_database(POSTGRES_SERVER_BACKUP)
    yield storage.psql
    storage.psql.backup_database(POSTGRES_SERVER_BACKUP)
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
    yield ptl.FractalClient("hpc3-22-03:7777", verify=False)
    # yield ptl.FractalClient(fractal_server)


@pytest.fixture(scope="function")
def empty_client():
    return FractalSnowflakeHandler().client()


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
