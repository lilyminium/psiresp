from typing import Any, Iterable, Tuple, List
import concurrent.futures

import numpy as np
import numpy.typing as npt

from . import exceptions

def wait_or_quit(futures: List[concurrent.futures.Future] = [],
                 timeout: Optional[float] = None,
                 command_log: str = "commands.log"):
    concurrent.futures.wait(futures, timeout=timeout)
    try:
        for future in futures:
            future.result()
    except exceptions.NoQMExecutionError:
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


def get_sin_cos_angle_oab(a: float, b: float) -> Tuple[float, float]:
    """
    Computes the angle between origin -- a -- b and
    returns the sine and cosine transforms

    Parameters
    ----------
    a: float
    b: float

    Returns
    -------
    sine_angle: float
    cosine_angle: float
    """
    hypotenuse = np.sqrt(a**2 + b**2)
    adjacent = abs(a)
    angle = np.arccos(adjacent/hypotenuse)

    if b >= 0:
        if a < 0:
            angle = np.pi - angle
    else:
        if a >= 0:
            angle = 2 * np.pi - angle
        else:
            angle = np.pi + angle

    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    return sin_angle, cos_angle


def rotate_x(n: int, coordinates: npt.NDArray):
    """
    Rotate coordinates such that the ``n``th coordinate of
    ``coordsinates`` becomes the x-axis. This is done *in-place*.

    Adapted from R.E.D. in perl.

    Parameters
    ----------
    n: int
        index of coordinates
    coordinates: numpy.ndarray
        coordinates
    """
    _, y, z = coordinates[n]
    sin_angle, cos_angle = get_sin_cos_angle_oab(y, z)
    ys = coordinates[:, 1].copy()
    zs = coordinates[:, 2].copy()
    coordinates[:, 1] = zs*sin_angle + ys*cos_angle
    coordinates[:, 2] = zs*cos_angle - ys*sin_angle


def rotate_z(n: int, coordinates: npt.NDArray):
    """
    Rotate coordinates such that the ``n``th coordinate of
    ``coordsinates`` becomes the z-axis. This is done *in-place*.

    Adapted from R.E.D. in perl.

    Parameters
    ----------
    n: int
        index of coordinates
    coordinates: numpy.ndarray
        coordinates
    """
    x, y, _ = coordinates[n]
    sin_angle, cos_angle = get_sin_cos_angle_oab(x, y)
    xs = coordinates[:, 0].copy()
    ys = coordinates[:, 1].copy()
    coordinates[:, 0] = xs*cos_angle + ys*sin_angle
    coordinates[:, 1] = ys*cos_angle - xs*sin_angle


def orient_rigid(i: int,
                 j: int,
                 k: int,
                 coordinates: npt.NDArray,
                 ) -> npt.NDArray:
    """
    Rigid-body reorientation such that the ``i`` th coordinate
    is the new origin; the ``j` `th coordinate defines the new
    x-axis; and the ``k`` th coordinate defines the XY plane.

    ``i``, ``j``, and ``k`` should all be different. They are
    indexed from 0.

    Adapted from R.E.D. in perl.

    Parameters
    ----------
    i: int
        index. Must be different to ``j`` and ``k``
    j: int
        index. Must be different to ``i`` and ``k``
    k: int
        index. Must be different to ``i`` and ``j``
    coordinates: numpy.ndarray of shape (N, 3)
        coordinates

    Returns
    -------
    coordinates: numpy.ndarray of shape (N, 3)
        New re-oriented coordinates
    """
    xyz = coordinates.copy()
    vec = coordinates[i]
    xyz -= vec
    rotate_x(j, xyz)
    rotate_z(j, xyz)
    rotate_x(k, xyz)
    return xyz


def rotate_rigid(i: int,
                 j: int,
                 k: int,
                 coordinates: npt.NDArray,
                 ) -> npt.NDArray:
    """
    Rigid-body rotation such that the ``i`` th and ``j`` th coordinate
    define a vector parallel to the x-axis; and the ``k`` th coordinate
    defines a plane parallel to the XY plane.

    ``i`` , ``j`` , and ``k`` should all be different. They are
    indexed from 0.

    Adapted from R.E.D. in perl.

    Parameters
    ----------
    i: int
        index. Must be different to ``j`` and ``k``
    j: int
        index. Must be different to ``i`` and ``k``
    k: int
        index. Must be different to ``i`` and ``j``
    coordinates: ndarray
        coordinates

    Returns
    -------
    coordinates: numpy.ndarray
        New rotated coordinates
    """
    return coordinates[i] + orient_rigid(i, j, k, coordinates)
