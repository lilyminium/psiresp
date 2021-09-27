import pathlib
import time
import shutil
import os

from qcfractal import FractalSnowflake, FractalSnowflakeHandler
import qcfractal.interface as ptl
import qcelemental as qcel

from psiresp.testing import TemporaryPostgres
from psiresp.tests.datafiles import DMSO
from qcfractal.postgres_harness import PostgresHarness


def reset_server_database(server):
    """Resets the server database for testing."""
    if "QCFRACTAL_RESET_TESTING_DB" in os.environ:
        server.storage._clear_db(server.storage._project_name)

    server.storage._delete_DB_data(server.storage._project_name)

    # Force a heartbeat after database clean if a manager is present.
    if server.queue_socket:
        server.await_results()


def postgres_server():

    storage = None
    psql = PostgresHarness({"database": {"port": 5432}})
    # psql = PostgresHarness({"database": {"port": 5432, "username": "qcarchive", "password": "mypass"}})
    if not psql.is_alive():
        print()
        print(
            f"Could not connect to a Postgres server at {psql.config.database_uri()}, this will increase time per test session by ~3 seconds."
        )
        print()
        storage = TemporaryPostgres()
        psql = storage.psql
        print("Using Database: ", psql.config.database_uri())

    return psql


def fractal_compute_server():
    """
    A FractalServer with a local Pool manager.
    """

    # Storage name
    storage_name = "test_qcfractal_compute_snowflake"
    pserver = postgres_server()
    pserver.create_database(storage_name)

    server = FractalSnowflake(
        max_workers=1,
        storage_project_name=storage_name,
        storage_uri=pserver.database_uri(),
        reset_database=True,
        start_server=False,
    )
    reset_server_database(server)
    return server


TMP_POSTGRES_SERVER = pathlib.Path("/Users/lily/pydev/psiresp/psiresp/tests/data/tmp_server")

# molecule = qcel.models.Molecule.from_file(DMSO)
molecule = ptl.Molecule.from_data("""
O 0 0 0
H 0 0 2
H 0 2 0
units bohr
""")

# storage = TemporaryPostgres()
# storage.psql.create_database("asdf")

# storage = TemporaryPostgres(database_name="test_psiresp",
#                             tmpdir=TMP_POSTGRES_SERVER)

# tmp_server = FractalSnowflake(max_workers=2,
#                               storage_project_name="test_psiresp",
#                               storage_uri=storage.psql.database_uri(),
#                               reset_database=False,
#                               start_server=False)
# tmp_client = tmp_server.client()

# with FractalSnowflake(max_workers=2, storage_uri=storage.psql.database_uri(),
#                       storage_project_name="asdf", start_server=False, reset_database=True) as server:
# print(server)
# tmp_client = server.client()
if __name__ == "__main__":
    storage_name = "test_qcfractal_compute_snowflake"
    pserver = postgres_server()
    pserver.create_database(storage_name)

    with FractalSnowflake(
        max_workers=1,
        storage_project_name=storage_name,
        storage_uri=pserver.database_uri(),
        reset_database=True,
        start_server=False,
    ) as tmp_server:
        reset_server_database(tmp_server)

        tmp_client = ptl.FractalClient(tmp_server)
        print(tmp_client)

        response = tmp_client.add_compute(program="psi4",
                                          basis="6-31g",
                                          method="b3lyp",
                                          driver="energy",
                                          molecule=molecule)
        print(response)

        tmp_server.await_results()
        complete = False
        while not complete:
            time.sleep(10)
            print("loop")
            n_complete = tmp_client.query_results(id=response.submitted)
            print(n_complete)
            if n_complete[0].status == "COMPLETE":
                complete = True

        print(response.ids)
        record = tmp_client.query_results(response.ids)
        print(record)
