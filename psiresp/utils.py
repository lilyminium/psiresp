from typing import Any, Iterable, Tuple, List, Callable
import concurrent.futures
import re

import numpy as np
import numpy.typing as npt

BOHR_TO_ANGSTROM = 0.52917721092
ANGSTROM_TO_BOHR = 1/BOHR_TO_ANGSTROM


class NoQMExecutionError(RuntimeError):
    pass


def load_data(path: Path) -> Data:
    """
    Parameters
    ----------
    path: pathlib.Path or str
        Data path

    Returns
    -------
    data: numpy.ndarray or pd.DataFrame
        numpy.ndarray or pd.DataFrame
    """
    path = str(path)
    suffix = pathlib.Path(path).suffix

    if suffix == "csv":
        loader = utils.read_csv
    elif suffix in ("dat", "txt"):
        loader = np.loadtxt
    elif suffix in ("npy", "npz"):
        loader = np.load
    elif suffix in ("xyz", "pdb", "mol2"):
        loader = utils.load_text
    else:
        raise ValueError(f"Can't find loader for {suffix} file")

    return loader(path)


def save_data(data: Data, path: Path):
    """
    Parameters
    ----------
    data: numpy.ndarray or pd.DataFrame
    path: pathlib.Path or str
        Filename
    """
    suffix = pathlib.Path(path).suffix

    if suffix == "csv":
        data.to_csv(path)
    elif suffix in ("dat", "txt"):
        np.savetxt(path, data)
    elif suffix == "npy":
        np.save(path, data)
    elif suffix == "npz":
        np.savez(path, **data)
    elif suffix == "xyz":
        if isinstance(data, str):
            with open(path, "w") as f:
                f.write(data)
        else:
            data.save_xyz_file(path, True)
    else:
        raise ValueError(f"Can't find saver for {suffix} file")
    logger.info(f"Saved to {os.path.abspath(path)}")


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
        from .options.qm import command_stream
        with open(command_log, "w") as f:
            f.write(command_stream.getvalue())
        raise SystemExit("Exiting to allow you to run QM jobs. "
                         f"Check {command_log} for required commands")


def is_iterable(obj: Any) -> bool:
    """Returns ``True`` if `obj` can be iterated over and is *not* a string
    nor a :class:`NamedStream`

    Adapted from MDAnalysis.lib.util.iterable
    """
    if isinstance(obj, str):
        return False
    if hasattr(obj, "__next__") or hasattr(obj, "__iter__"):
        return True
    try:
        len(obj)
    except (TypeError, AttributeError):
        return False
    return True


def as_iterable(obj: Any) -> Iterable:
    """Returns `obj` so that it can be iterated over.

    A string is *not* considered an iterable and is wrapped into a
    :class:`list` with a single element.

    See Also
    --------
    is_iterable
    """
    if not is_iterable(obj):
        obj = [obj]
    return obj


def split_docstring_into_parts(docstring: str) -> Dict[str, List[str]]:
    """Split docstring around headings"""
    parts = defaultdict(list)
    heading_pattern = "[ ]{4}[A-Z][a-z]+\s*\n[ ]{4}[-]{4}[-]+\s*\n"
    directive_pattern = "[ ]{4}\.\. [a-z]+:: .+\n"
    pattern = re.compile("(" + heading_pattern + "|" + directive_pattern + ")")
    sections = re.split(pattern, docstring)
    parts["base"] = sections.pop(0)
    while sections:
        heading_match, section = sections[:2]
        sub_pattern = "([A-Z][a-z]+|[ ]{4}\.\. [a-z]+:: .+\n)"
        heading = re.search(sub_pattern, heading_match).groups()[0]
        section = heading_match + section
        parts[heading] = section.split("\n")
        sections = sections[2:]
    return parts


def join_split_docstring(parts: Dict[str, List[str]]) -> str:
    """Join split docstring back into one string"""
    docstring = parts.pop("base", "")
    headings = ("Parameters", "Attributes", "Examples")
    for heading in headings:
        section = parts.pop(heading, [])
        docstring += "\n".join(section)
    for section in parts.values():
        docstring += "\n".join(section)
    return docstring


def extend_docstring_with_base(docstring: str, base_class: type) -> str:
    """Extend docstring with the parameters in `base_class`"""
    doc_parts = split_docstring_into_parts(docstring)
    base_parts = split_docstring_into_parts(base_class.__doc__)
    headings = ("Parameters", "Attributes", "Examples")
    for k in headings:
        if k in base_parts:
            section = base_parts.pop(k)
            if doc_parts.get(k):
                section = section[2:]
            doc_parts[k].extend(section)

    for k, lines in base_parts.items():
        if k != "base" and k in doc_parts:
            doc_parts[k].extend(lines[2:])

    return join_split_docstring(doc_parts)
