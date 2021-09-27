import numpy as np
import scipy.sparse
import qcelemental as qcel

from . import base


def array_ops(func):
    def wrapper(self, other):
        try:
            arr = func(self.matrix, other.array)
        except AttributeError:
            arr = func(self.matrix, other)
        if arr is not None:
            return type(self)(arr)
    return wrapper


class ConstraintMatrix(base.Model):
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
        return cls(np.zeros((n_dim + 1, n_dim)))

    @classmethod
    def from_orientations(cls, orientations=[],
                          boltzmann_weight: bool = False,
                          temperature: float = 298.15):

        if not len(orientations):
            raise ValueError("At least one Orientation must be provided")

        if not all(o.esp is not None for o in orientations):
            raise ValueError("All Orientations must have had the ESP computed")

        first = orientations[0]
        matrix = cls.with_n_dim(first.qcmol.geometry.shape[0])
        for orient in orientations:
            mat = orient.constraint_matrix
            if boltzmann_weight:
                mat *= (orient.get_boltzmann_weight(temperature) ** 2)
            matrix += mat
        return matrix

    @classmethod
    def from_a_and_b(cls, a, b):
        n_dim = a.shape[0]
        array = np.empty((n_dim + 1, n_dim))
        array[:1] = a
        array[-1] = b
        return cls(array)

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

    @property
    def b(self):
        return self.matrix[-1]


class SparseConstraintMatrix:

    a: ...
    b: ...

    @classmethod
    def from_molecules(cls, molecules=[], **kwargs):
        matrices = [
            ConstraintMatrix.from_orientations(
                orientations=[o for conf in mol.conformers for o in conf.orientations],
                **kwargs
            )
            for mol in molecules
        ]

        a_coo = [scipy.sparse.coo_matrix(mat.a) for mat in matrices]
        a = scipy.sparse.block_diag(a_coo).tocsr()
        b = np.concatenate([mat.b for mat in matrices])

        all_atom_rows = scipy.sparse.block_diag(
            [scipy.sparse.coo_matrix(np.ones(m.n_atoms)) for m in molecules]
        )
        A = scipy.sparse.bmats([[a, rows.T], [rows, None]]).tocsr()

        charges = [m.charge for m in molecules]
        B = scipy.sparse.csr_matrix(np.r_[b, charges])

        return cls(a=A, b=B)

    def __init__(self, molecules, mask=None):
        self._set_constraint_matrices(molecules)
        if mask is None:
            mask = []
        self._array_mask = mask
        self._array_indices = np.where(mask)[0]

        self._charges = None
        self._previous_charges = None
        self._previous_a = None

    @property
    def a(self):
        return self._a[(self._array_indices, self._array_indices)]

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
        return np.abs(self._charges - self._previous_charges).max()

    def _set_constraint_matrices(self, molecules):
        matrices = [
            ConstraintMatrix.from_orientations(
                orientations=[o for conf in mol.conformers for o in conf.orientations],
                **kwargs
            )
            for mol in molecules
        ]

        a_coo = [scipy.sparse.coo_matrix(mat.a) for mat in matrices]
        a = scipy.sparse.block_diag(a_coo).tocsr()
        b = np.concatenate([mat.b for mat in matrices])

        all_atom_rows = scipy.sparse.block_diag(
            [scipy.sparse.coo_matrix(np.ones(m.n_atoms)) for m in molecules]
        )
        sparse_input = [[a, rows.T], [rows, None]]
        self._original_a = self._a = scipy.sparse.bmats(sparse_input).tocsr()

        charges = [m.charge for m in molecules]
        self._b = scipy.sparse.csr_matrix(np.r_[b, charges])

    def _solve(self):
        self._previous_charges = self._charges
        try:
            self._charges = scipy.sparse.linalg.spsolve(self._a, self._b)
        except RuntimeError as e:  # TODO: this could be slow?
            self._charges = scipy.sparse.linalg.lsmr(self._a, self._b)

    def _iter_solve(self, a_array, b_squared):
        self._a = self._original_a.copy()
        self.a += a_array / np.sqrt(self.charges ** 2 + b_squared)
        self._solve()
