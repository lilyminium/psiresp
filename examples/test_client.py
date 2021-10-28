import pathlib
import time
import shutil
import os

import numpy as np

from qcfractal import FractalSnowflake, FractalSnowflakeHandler
import qcfractal.interface as ptl
import qcelemental as qcel

from psiresp.testing import TemporaryPostgres
from psiresp.tests.datafiles import DMSO, DMSO_O1, DMSO_O2
from qcfractal.postgres_harness import PostgresHarness


mols = [qcel.models.Molecule.from_file(x)
        for x in [DMSO, DMSO_O1, DMSO_O2]]
# molecule = ptl.Molecule.from_data("""
# O 0 0 0
# H 0 0 2
# H 0 2 0
# units bohr
# """)

if __name__ == "__main__":
    storage = TemporaryPostgres(database_name="test_psiresp")

    # storage_name = "test_qcfractal_compute_snowflake"
    # pserver = postgres_server()
    # pserver.create_database(storage_name)

    with FractalSnowflake(
        max_workers=1,
        storage_project_name="test_psiresp",
        storage_uri=storage.psql.database_uri(),
        reset_database=False,
        start_server=True,
    ) as tmp_server:
        # reset_server_database(tmp_server)

        tmp_client = ptl.FractalClient(tmp_server)

        wfn_protocols = {"wavefunction": "orbitals_and_eigenvalues"}
        response = tmp_client.add_compute(program="psi4",
                                          basis="6-31g*",
                                          method="hf",
                                          driver="energy",
                                          molecule=mols,
                                          protocols=wfn_protocols)

        # tmp_server.await_results()
        complete = False
        while not complete:
            time.sleep(10)
            print("loop")
            n_complete = tmp_client.query_results(id=response.submitted)
            print(n_complete)
            # print(tmp_client.query_managers())
            if n_complete[0].status == "COMPLETE":
                complete = True

        print(response.ids)
        # record = tmp_client.query_results(id=["1"])  # response.ids)
        # print(record)

    storage.psql.backup_database("/Users/lily/pydev/psiresp/psiresp/tests/data/database.psql")
