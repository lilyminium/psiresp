#!/usr/bin/env python


import pathlib
import time
from typing import Optional
import tempfile
import pathlib
import glob
import shutil
import subprocess
import shutil
import os

from qcfractal.postgres_harness import find_port, FractalConfig, atexit
# from qcfractal.postgres_harness import PostgresHarness as QCPostgresHarness
# from qcfractal.testing import TemporaryPostgres as QCTemporaryPostgres
from psiresp.testing import PostgresHarness, TemporaryPostgres


from qcfractal.interface.models import ResultRecord
from qcfractal.storage_sockets import storage_socket_factory
from qcfractal import FractalSnowflake, FractalSnowflakeHandler
import qcfractal.interface as ptl
import qcelemental as qcel


POSTGRES_SERVER_BACKUP = "database.psql"


if __name__ == "__main__":
    storage = TemporaryPostgres(database_name="test_psiresp")
    storage.psql.restore_database(POSTGRES_SERVER_BACKUP)

    output_json = glob.glob("output/*.json")
    print(storage)
    records = []
    for file in output_json:
        try:
            records.append(ResultRecord.parse_file(file))
        except:
            print(file)

    print(len(records))

    with FractalSnowflake(
        max_workers=1,
        storage_project_name="test_psiresp",
        storage_uri=storage.psql.database_uri(),
        reset_database=False,
        start_server=True,
    ) as tmp_server:
        print(tmp_server)
        socket = storage_socket_factory(tmp_server._storage_uri,
                                        project_name="test_psiresp",
                                        skip_version_check=True,
                                        )
        socket.add_results(records)
        print(socket)

    storage.psql.backup_database(POSTGRES_SERVER_BACKUP)
    print(f"Saved to {POSTGRES_SERVER_BACKUP}")
