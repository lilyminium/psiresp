from typing import List, Callable, Optional
import concurrent.futures

from ..mixins.qm import command_stream


class NoQMExecutionError(RuntimeError):
    """Special error to tell job to quit if there are QM jobs to run"""


def run_with_executor(functions: List[Callable] = [],
                      executor: Optional[concurrent.futures.Executor] = None,
                      timeout: Optional[float] = None,
                      command_log: str = "commands.log"):
    """Submit ``functions`` to potential ``executor``, or run in serial

    Parameters
    ----------
    functions: list of functions
        List of functions to run
    executor: concurrent.futures.Executor (optional)
        If given, the functions will be submitted to this executor.
        If not, the functions will run in serial.
    timeout: float
        Timeout for waiting for the executor to complete
    command_log: str
        File to write commands to, if there are QM jobs to run
    """
    futures = []
    for func in functions:
        try:
            future = executor.submit(func)
        except AttributeError:
            func()
        else:
            futures.append(future)
    wait_or_quit(futures, timeout=timeout, command_log=command_log)


def wait_or_quit(futures: List[concurrent.futures.Future] = [],
                 timeout: Optional[float] = None,
                 command_log: str = "commands.log"):
    """Either wait for futures to complete, or quit

    Parameters
    ----------
    futures: list of futures
        Futures to complete
    timeout: float
        Timeout for waiting for the executor to complete
    command_log: str
        File to write commands to, if there are QM jobs to run

    Raises
    ------
    SystemExit
        if there are QM jobs to run
    """
    concurrent.futures.wait(futures, timeout=timeout)
    try:
        for future in futures:
            future.result()
    except NoQMExecutionError:
        with open(command_log, "w") as f:
            f.write(command_stream.getvalue())
        raise SystemExit("Exiting to allow you to run QM jobs. "
                         f"Check {command_log} for required commands")
