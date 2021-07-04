from typing import List, Callable, Optional
import concurrent.futures

from ..mixins.qm import command_stream


class NoQMExecutionError(RuntimeError):
    pass


def run_with_executor(functions: List[Callable] = [],
                      executor: Optional[concurrent.futures.Executor] = None,
                      timeout: Optional[float] = None,
                      command_log: str = "commands.log"):
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
    concurrent.futures.wait(futures, timeout=timeout)
    try:
        for future in futures:
            future.result()
    except NoQMExecutionError:
        with open(command_log, "w") as f:
            f.write(command_stream.getvalue())
        raise SystemExit("Exiting to allow you to run QM jobs. "
                         f"Check {command_log} for required commands")
