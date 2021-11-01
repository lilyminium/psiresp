#!/usr/bin/env python

from qcfractal.testing import TemporaryPostgres as QCTemporaryPostgres
from qcfractal.postgres_harness import PostgresHarness as QCPostgresHarness
from qcfractal.postgres_harness import find_port, FractalConfig, atexit
import subprocess
import shutil
import tempfile
from typing import Optional
import argparse
import time
import pathlib
import numpy as np

from qcfractal import FractalSnowflakeHandler, FractalSnowflake, FractalServer
import qcelemental as qcel
import qcengine as qcng
from qcfractal import interface as ptl
from qcfractal.interface.models.records import RecordStatusEnum

parser = argparse.ArgumentParser("Run resp")
parser.add_argument("infile", type=str)
parser.add_argument("--basis", type=str, default="6-31g*")
parser.add_argument("--method", type=str, default="hf")
parser.add_argument("--state", type=str, default="gas")

PCM_KEYWORDS = {
    "pcm": "true",
    "pcm_scf_type": "total",
    "pcm__input": r"""
        Units = Angstrom
        Medium {
            SolverType = CPCM
            Solvent = water
        }

        Cavity {
            RadiiSet = Bondi # Bondi | UFF | Allinger
            Type = GePol
            Scaling = True # radii for spheres scaled by 1.2
            Area = 0.3
            Mode = Implicit
        }
        """
}


class PostgresHarness(QCPostgresHarness):
    def _run(self, commands):
        command_str = " ".join(list(map(str, commands)))
        if (any(x in command_str for x in ["-p ", "--port="])
                and not any(x in command_str for x in ["-h ", "--host="])):
            commands.extend(["-h", "127.0.0.1"])

        proc = subprocess.run(commands, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout = proc.stdout.decode()
        if not self.quiet:
            self.logger(stdout)

        ret = {"retcode": proc.returncode, "stdout": stdout, "stderr": proc.stderr.decode()}
        return ret

    def restore_database(self, filename) -> None:

        # Reasonable check here
        self._check_psql()

        self.create_database(self.config.database.database_name)

        # fmt: off
        cmds = [
            shutil.which("pg_restore"),
            "-c",
            f"--port={self.config.database.port}",
            f"--dbname={self.config.database.database_name}",
            filename
        ]
        # fmt: on

        self.logger(f"pg_backup command: {'  '.join(cmds)}")
        ret = self._run(cmds)

        if ret["retcode"] != 0:
            self.logger(ret["stderr"])
            raise ValueError("\nFailed to restore the database.\n")


class TemporaryPostgres(QCTemporaryPostgres):
    def __init__(
        self,
        database_name: Optional[str] = None,
        tmpdir: Optional[str] = None,
        quiet: bool = True,
        logger: "print" = print,
    ):
        """A PostgreSQL instance run in a temporary folder.

        ! Warning ! All data is lost when this object is deleted.

        Parameters
        ----------
        database_name : Optional[str], optional
            The database name to create.
        tmpdir : Optional[str], optional
            A directory to create the postgres instance in, if not None the data is not deleted upon shutdown.
        quiet : bool, optional
            If True, does not log any operations
        logger : print, optional
            The logger to show the operations to.
        """

        self._active = True

        if not tmpdir:
            self._db_tmpdir = tempfile.TemporaryDirectory().name
        else:
            tmpdir = pathlib.Path(tmpdir)
            self._db_tmpdir = str(tmpdir.absolute())

        self.quiet = quiet
        self.logger = logger

        config_data = {"port": find_port(), "directory": self._db_tmpdir}
        if database_name:
            config_data["database_name"] = database_name
        self.config = FractalConfig(database=config_data)
        self.psql = QCPostgresHarness(self.config)
        self.psql.initialize_postgres()
        self.psql.init_database()
        # self.psql.start()

        atexit.register(self.stop)

# Update database
# FractalServer.storage = SQLAlchemySocket
# FractalServer.storage.add_results([ResultRecords])


if __name__ == "__main__":
    args = parser.parse_args()
    # mol = qcel.models.Molecule.from_file(args.infile)
    # mol = qcel.models.Molecule(**{"symbols": ["He"], "geometry": [0, 0, 0]})
    mol = ptl.Molecule.from_data("""
        O 0 0 0
        H 0 0 2
        H 0 2 0
        units bohr
        """)

    print(mol)

    storage = QCTemporaryPostgres(database_name="test_psiresp")

    with FractalSnowflake(
        # max_workers=4,
        storage_project_name="test_psiresp",
        storage_uri=storage.psql.database_uri(),
        # reset_database=True,
        # start_server=False,
    ) as server:
        # with FractalSnowflakeHandler() as server:
        print(server)

        client = ptl.FractalClient(server, verify=False)
        print(client)

        # spec = {
        #     "keywords": None,
        #     "qc_spec": {
        #         "driver": "energy",
        #         "method": "b3lyp",
        #         "basis": "6-31g",
        #         "program": "psi4"
        #     },
        # }

        # # Ask the server to compute a new computation
        # response = client.add_procedure("optimization", "geometric", spec, [mol])

        # print(server.await_results())

        # complete = False
        # while not complete:
        #     time.sleep(10)
        #     print("loop")
        #     n_complete = client.query_procedures(id=response.ids)
        #     print(client.query_managers())
        #     # print(server.list_current_tasks())
        #     # print(server.check_manager_heartbeats())
        #     # print(server.list_managers())
        #     print(n_complete)
        #     # print(tmp_client.query_managers())
        #     if n_complete[0].status == "COMPLETE":
        #         complete = True

        keywords = {"maxiter": 300}
        keyword_id = client.add_keywords([ptl.models.KeywordSet(values=keywords)])[0]

        wfn_protocols = {"wavefunction": "orbitals_and_eigenvalues"}

        computation = dict(
            program="psi4",
            basis=args.basis,
            method="b3lyp",
            driver="energy",
            molecule=[mol],
            # keywords=keyword_id,
            # protocols=wfn_protocols,
        )

        print(computation)

        response = client.add_compute(**computation)
        print(response)

        print(server.await_results())
        # print("awaited")
        # print(client.query_managers())
        complete = False
        while not complete:
            time.sleep(10)
            print("loop")
            n_complete = client.query_results(id=response.submitted)
            print(client.query_managers())
            # print(server.list_current_tasks())
            # print(server.check_manager_heartbeats())
            # print(server.list_managers())
            print(n_complete)
            # print(tmp_client.query_managers())
            if n_complete[0].status == "COMPLETE":
                complete = True

        print(response.ids)

    #     ret = client.query_results(response.ids)[0]
    # print(ret)

    # storage = TemporaryPostgres(database_name="test_psiresp")
    # print(storage)

    # with FractalSnowflake(
    #     max_workers=1,
    #     storage_project_name="test_psiresp",
    #     storage_uri=storage.psql.database_uri(),
    #     reset_database=False,
    #     start_server=True,
    # ) as server:
    #     print(server)
    #     client = ptl.FractalClient(server)
    #     print(client)

    #     keywords = {"maxiter": 300}
    #     if args.state != "gas":
    #         keywords.update(PCM_KEYWORDS)

    #     keyword_id = client.add_keywords([ptl.models.KeywordSet(values=keywords)])[0]
    #     print(keyword_id)

    #     wfn_protocols = {"wavefunction": "orbitals_and_eigenvalues"}

    #     computation = dict(
    #         program="psi4",
    #         basis=args.basis,
    #         method=args.method,
    #         driver="energy",
    #         molecule=[mol],
    #         keywords=keyword_id,
    #         protocols=wfn_protocols,
    #     )
    #     print(computation)
    #     response = client.add_compute(**computation)

    #     print(response)
    #     ret = client.query_results(response.ids)[0]
    #     print(ret)
    #     server.await_results()
    #     print("awaited")
    #     ret = client.query_results(response.ids)[0]

    molname = pathlib.Path(args.infile).stem

    if args.method.upper() == "PW6B95":
        suffix = "_resp2"
    else:
        suffix = ""

    jsonfile = f"output/{molname}{suffix}_{args.state}_energy.json"
    with open(jsonfile, "w") as f:
        f.write(ret.json())

    print(f"Wrote to {jsonfile}")
