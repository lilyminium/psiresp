from typing import Optional, Tuple
import copy

import numpy as np
import scipy.sparse
import scipy.sparse.linalg

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


class ESPSurfaceConstraintMatrix(base.Model):

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

        if not all(ort.esp is not None for ort in orientations):
            raise ValueError("All Orientations must have had the ESP computed")

        first = orientations[0]
        matrix = cls.with_n_dim(first.qcmol.geometry.shape[0])
        for ort in orientations:
            matrix += ort.get_weighted_matrix(temperature=temperature)
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


class SparseGlobalConstraintMatrix(base.Model):

    a: scipy.sparse.csr_matrix
    b: np.ndarray
    n_structure_array: Optional[np.ndarray] = None
    mask: Optional[np.ndarray] = None

    _original_a: Optional[scipy.sparse.csr_matrix] = None
    _charges: Optional[np.ndarray] = None
    _previous_charges: Optional[np.ndarray] = None
    _n_atoms: Optional[int] = None
    _array_mask: Optional[Tuple[np.ndarray, np.ndarray]] = None
    _array_indices: Optional[np.ndarray] = None

    @classmethod
    def from_constraints(cls, surface_constraints,
                         charge_constraints,
                         exclude_hydrogens=True):
        a = scipy.sparse.csr_matrix(surface_constraints.a)
        b = surface_constraints.b

        if charge_constraints.n_constraints:
            a_block = scipy.sparse.hstack([
                scipy.sparse.coo_matrix(dense)
                for dense in charge_constraints.to_a_col_constraints()
            ])
            b_block_ = charge_constraints.to_b_constraints()
            a = scipy.sparse.bmat(
                [[a, a_block],
                 [a_block.transpose(), None]],
                format="csr",
            )
            b_block = np.zeros(a_block.shape[1])
            b_block[:len(b_block_)] = b_block_
            b = np.r_[b, b_block]

        n_structure_array = np.concatenate(
            [[mol.n_orientations] * mol.n_atoms
             for mol in charge_constraints.molecules]
        )

        symbols = np.concatenate(
            [mol.qcmol.symbols
             for mol in charge_constraints.molecules]
        )
        mask = np.ones_like(symbols, dtype=bool)
        if exclude_hydrogens:
            mask[np.where(symbols == "H")[0]] = False

        return cls(a=a, b=b, n_structure_array=n_structure_array, mask=mask)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._n_atoms = len(self.b)
        if self.n_structure_array is None:
            self.n_structure_array = np.ones(self._n_atoms)
        if self.mask is None:
            self.mask = np.ones(self._n_atoms, dtype=bool)
        self._array_indices = np.where(self.mask)[0]
        diag_indices = np.diag_indices(self._n_atoms)
        self._array_mask = (diag_indices[0][self._array_indices],
                            diag_indices[1][self._array_indices])
        self._original_a = copy.deepcopy(self.a)

    @property
    def charge_difference(self):
        if self._previous_charges is None or self._charges is None:
            return np.inf
        return np.max(np.abs(self._charges - self._previous_charges)[:self._n_atoms])

    @property
    def charges(self):
        if self._charges is None:
            return None
        return self._charges[self._array_indices]

    def _solve(self):
        self._previous_charges = copy.deepcopy(self._charges)
        try:
            self._charges = scipy.sparse.linalg.spsolve(self.a, self.b)
        except Warning:
            breakpoint()
        except RuntimeError as e:  # TODO: this could be slow?
            self._charges = scipy.sparse.linalg.lsmr(self.a, self.b)[0]
        else:
            if np.isnan(self._charges[0]):
                self._charges = scipy.sparse.linalg.lsmr(self.a, self.b)[0]

    def _iter_solve(self, resp_a, resp_b, b2):
        hyp_a = (resp_a * self.n_structure_array)[self._array_indices]
        increment = hyp_a / np.sqrt(self._charges[self._array_indices] ** 2 + b2)
        self.a = self._original_a.copy()

        a_shape = self.a[self._array_mask].shape
        self.a[self._array_mask] += increment.reshape(a_shape)
        self._solve()
