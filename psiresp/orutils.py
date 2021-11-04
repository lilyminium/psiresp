from typing import Tuple, List
import itertools

import numpy as np


# def id_to_indices(atom_ids: List[int]) -> List[int]:
#     """Convert atom numbers (indexed from 1) to indices (indexed from 0)

#     This also works with negative atom numbers, where -1 is the last item.

#     Parameters
#     ----------
#     atom_ids: list of ints
#     """
#     return [a - 1 if a > 0 else a for a in atom_ids]


def generate_atom_combinations(symbols: List[str]):
    """Yield combinations of atom indices for transformations

    The method first yields combinations of 3 heavy atom indices.
    Each combination is followed by its reverse. Once the heavy atoms
    are exhausted, the heavy atoms then get combined with the hydrogens.

    Parameters
    ----------
    symbols: list of str
        List of atom elements

    Examples
    --------

    ::

        >>> symbols = ["H", "C", "C", "O", "N"]
        >>> comb = OrientationOptions.generate_atom_combinations(symbols)
        >>> next(comb)
        (1, 2, 3)
        >>> next(comb)
        (3, 2, 1)
        >>> next(comb)
        (1, 2, 4)
        >>> next(comb)
        (4, 2, 1)

    """
    symbols = np.asarray(symbols)
    is_H = symbols == "H"
    h_atoms = list(np.flatnonzero(is_H))
    heavy_atoms = list(np.flatnonzero(~is_H))
    seen = set()

    for comb in itertools.combinations(heavy_atoms, 3):
        seen.add(comb)
        yield comb
        yield comb[::-1]

    for comb in itertools.combinations(heavy_atoms + h_atoms, 3):
        if comb in seen:
            continue
        seen.add(comb)
        yield comb
        yield comb[::-1]


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


def rotate_x(n: int, coordinates: np.ndarray):
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


def rotate_z(n: int, coordinates: np.ndarray):
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


def rigid_orient(i: int,
                 j: int,
                 k: int,
                 coordinates: np.ndarray,
                 ) -> np.ndarray:
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


def rigid_rotate(i: int,
                 j: int,
                 k: int,
                 coordinates: np.ndarray,
                 ) -> np.ndarray:
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
    return coordinates[i] + rigid_orient(i, j, k, coordinates)
