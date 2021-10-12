
import copy

import numpy as np
import scipy.sparse
import scipy.sparse.linalg

from numpy.testing import assert_allclose
import qcelemental as qcel

from . import base


def array_ops(func):
    def wrapper(self, other):
        try:
            arr = func(self.matrix, other.matrix)
        except AttributeError:
            arr = func(self.matrix, other)
        if arr is not None:
            return type(self)(matrix=arr)
    return wrapper


class ConstraintMatrix(base.Model):

    # class Config(base.Model.Config):
    #     extra = "allow"

    matrix: np.ndarray

    __add__ = array_ops(np.ndarray.__add__)
    __sub__ = array_ops(np.ndarray.__sub__)
    __pow__ = array_ops(np.ndarray.__pow__)
    __mul__ = array_ops(np.ndarray.__mul__)
    __truediv__ = array_ops(np.ndarray.__truediv__)

    __iadd__ = array_ops(np.ndarray.__iadd__)
    __isub__ = array_ops(np.ndarray.__isub__)
    __ipow__ = array_ops(np.ndarray.__ipow__)
    __imul__ = array_ops(np.ndarray.__imul__)
    __itruediv__ = array_ops(np.ndarray.__itruediv__)

    @classmethod
    def with_n_dim(cls, n_dim: int):
        return cls(matrix=np.zeros((n_dim + 1, n_dim)))

    @classmethod
    def from_orientations(cls, orientations=[],
                          temperature: float = 298.15):

        if not len(orientations):
            raise ValueError("At least one Orientation must be provided")

        if not all((o._orientation_esp is not None
                    and o._orientation_esp.esp is not None)
                   for o in orientations):
            raise ValueError("All Orientations must have had the ESP computed")

        first = orientations[0]
        matrix = cls.with_n_dim(first.qcmol.geometry.shape[0])
        for ort in orientations:
            mat = ort._orientation_esp.get_weighted_matrix(temperature=temperature)
            matrix += mat
        return matrix

    @classmethod
    def from_a_and_b(cls, a, b):
        n_dim = a.shape[0]
        array = np.empty((n_dim + 1, n_dim))
        matrix = cls(matrix=array)
        matrix.a = a
        matrix.b = b

        return matrix

    def validate_array(cls, array):
        array = np.asarray(array).astype(float)

        assert len(array.shape) == 2
        if array.shape[1] > array.shape[0]:
            array = array.T

        if array.shape[1] != array.shape[0] + 1:
            raise ValueError("Array should be of shape N x N + 1")

        return array

    @property
    def n_dim(self):
        return self.matrix.shape[1]

    @property
    def a(self):
        return self.matrix[:-1]

    @a.setter
    def a(self, value):
        self.matrix[:-1] = value

    @property
    def b(self):
        return self.matrix[-1]

    @b.setter
    def b(self, value):
        self.matrix[-1] = value


class SparseConstraintMatrix:

    def __init__(self, a, b, n_atoms, mask=None):
        self._a = scipy.sparse.csr_matrix(a)
        self._b = b
        if mask is None:
            mask = np.ones(n_atoms, dtype=bool)

        self._array_indices = np.where(mask)[0]

        diag_indices = np.diag_indices(n_atoms)
        self._array_mask = (diag_indices[0][mask],
                            diag_indices[1][mask])

        self._n_atoms = n_atoms
        self._charges = None
        self._previous_charges = None
        self._previous_a = None
        self._original_a = copy.deepcopy(self._a)

    @property
    def a(self):
        return self._a[self._array_mask]

    @a.setter
    def a(self, value):
        self._a[(self._array_indices, self._array_indices)] = value

    @property
    def charges(self):
        if self._charges is None:
            return None
        return self._charges[self._array_indices]

    @property
    def previous_charges(self):
        if self._previous_charges is None:
            return None
        return self._previous_charges[self._array_indices]

    @property
    def charge_difference(self):
        if self._previous_charges is None or self._charges is None:
            return np.inf
        return np.max(np.abs(self._charges - self._previous_charges)[:self._n_atoms])

    def _solve(self):
        self._previous_charges = self._charges
        try:
            self._charges = scipy.sparse.linalg.spsolve(self._a, self._b)
        except RuntimeError as e:  # TODO: this could be slow?
            self._charges = scipy.sparse.linalg.lsmr(self._a, self._b)[0]

    def _iter_solve(self, a_array, b_squared):
        self._a = self._original_a.copy()
        self.a += a_array / np.sqrt(self.charges ** 2 + b_squared)
        # self.a += a_array / np.sqrt(self.charges ** 2 + b_squared)
        self._solve()
