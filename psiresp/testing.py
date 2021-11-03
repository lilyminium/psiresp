from typing import Optional, Union
import tempfile
import pathlib
import shutil
import subprocess

from qcfractal.postgres_harness import find_port, FractalConfig, atexit
from qcfractal.postgres_harness import PostgresHarness as QCPostgresHarness
from qcfractal.testing import TemporaryPostgres as QCTemporaryPostgres
from qcfractal import FractalSnowflake as QCFractalSnowflake


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

    def restore_database(self, filename, ) -> None:

        # Reasonable check here
        self._check_psql()

        self.create_database(self.config.database.database_name)

        # fmt: off
        cmds = [
            shutil.which("pg_restore"),
            "-c",
            f"--port={self.config.database.port}",
            f"--dbname={self.config.database.database_name}",
            "--no-privileges",
            "--no-owner",
            filename
        ]
        # fmt: on

        self.logger(f"pg_backup command: {'  '.join(cmds)}")
        ret = self._run(cmds)

        if ret["retcode"] != 0:
            self.logger(ret["stderr"])
            raise ValueError(ret["stderr"])
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
        self.psql = PostgresHarness(self.config)
        self.psql.initialize_postgres()
        self.psql.init_database()
        # self.psql.start()

        atexit.register(self.stop)


class FractalSnowflake(QCFractalSnowflake):

    def __init__(self, max_workers: Optional[int] = 2,
                 storage_project_name: str = "temporary_snowflake",
                 max_active_services: int = 20,
                 logging: Union[bool, str] = False,
                 start_server: bool = True,
                 reset_database: bool = False,):
        storage = TemporaryPostgres(database_name=storage_project_name)
        storage_uri = storage.database_uri(safe=False, database="")
        super().__init__(max_workers=max_workers,
                         max_active_services=max_active_services,
                         storage_uri=storage_uri,
                         storage_project_name=storage_project_name,
                         logging=logging,
                         start_server=start_server,
                         reset_database=reset_database)
        self._storage = storage
