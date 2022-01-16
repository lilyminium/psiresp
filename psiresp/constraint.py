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
    """
    Class used to iteratively solve the matrix of linear
    equations Ax=b for the charges. In this analogy,
    `A` is the coefficient matrix of summed inverse distances
    to each grid point, and charge constraints;
    `b` is the constant vector of summed potential at each
    grid point, and values of the charge constraints.
    We solve for x, the charges.

    Users should not need to use this class directly.
    """

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
    def from_orientations(cls, orientations=[], temperature: float = 298.15):

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
    def from_coefficient_matrix(cls, coefficient_matrix, constant_vector=None):
        n_dim = coefficient_matrix.shape[0]
        if constant_vector is None:
            constant_vector = np.zeros(n_dim)
        else:
            try:
                constant_vector = constant_vector.reshape((n_dim,))
            except ValueError:
                msg = f"`constant_vector` must have shape ({n_dim},)"
                raise ValueError(msg)
        placeholder = np.empty((n_dim + 1, n_dim))
        matrix = cls(matrix=placeholder)
        matrix.coefficient_matrix = coefficient_matrix
        matrix.constant_vector = constant_vector

        return matrix

    @property
    def n_dim(self):
        return self.matrix.shape[1]

    @property
    def coefficient_matrix(self):
        return self.matrix[:-1]

    @coefficient_matrix.setter
    def coefficient_matrix(self, value):
        self.matrix[:-1] = value

    @property
    def constant_vector(self):
        return self.matrix[-1]

    @constant_vector.setter
    def constant_vector(self, value):
        self.matrix[-1] = value


class SparseGlobalConstraintMatrix(base.Model):

    coefficient_matrix: scipy.sparse.csr_matrix
    constant_vector: np.ndarray
    n_structure_array: Optional[np.ndarray] = None
    mask: Optional[np.ndarray] = None

    _original_coefficient_matrix: Optional[scipy.sparse.csr_matrix] = None
    _charges: Optional[np.ndarray] = None
    _previous_charges: Optional[np.ndarray] = None
    _n_atoms: Optional[int] = None
    _array_mask: Optional[Tuple[np.ndarray, np.ndarray]] = None
    _array_indices: Optional[np.ndarray] = None

    @classmethod
    def from_constraints(
        cls,
        surface_constraints,
        charge_constraints,
        exclude_hydrogens: bool = True,
    ):
        a = scipy.sparse.csr_matrix(surface_constraints.coefficient_matrix)
        b = surface_constraints.constant_vector

        if charge_constraints.n_constraints:
            a_block = scipy.sparse.hstack([
                scipy.sparse.coo_matrix(dense)
                for dense in charge_constraints.to_a_col_constraints()
            ])
            b_block_ = charge_constraints.to_b_constraints()
            a = scipy.sparse.bmat(
                [[a, a_block], [a_block.transpose(), None]],
                format="csr",
            )
            b_block = np.zeros(a_block.shape[1])
            b_block[: len(b_block_)] = b_block_
            b = np.r_[b, b_block]

        n_structure_array = np.concatenate([
            [mol.n_orientations] * mol.n_atoms
            for mol in charge_constraints.molecules
        ])

        symbols = np.concatenate(
            [mol.qcmol.symbols for mol in charge_constraints.molecules]
        )
        mask = np.ones_like(symbols, dtype=bool)
        if exclude_hydrogens:
            mask[np.where(symbols == "H")[0]] = False

        return cls(
            coefficient_matrix=a,
            constant_vector=b,
            n_structure_array=n_structure_array,
            mask=mask,
        )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._n_atoms = len(self.constant_vector)
        if self.n_structure_array is None:
            self.n_structure_array = np.ones(self._n_atoms)
        if self.mask is None:
            self.mask = np.ones(self._n_atoms, dtype=bool)
        self._array_indices = np.where(self.mask)[0]
        diag_indices = np.diag_indices(self._n_atoms)
        self._array_mask = (
            diag_indices[0][self._array_indices],
            diag_indices[1][self._array_indices],
        )
        self._original_coefficient_matrix = copy.deepcopy(
            self.coefficient_matrix
        )

    @property
    def charge_difference(self):
        if self._previous_charges is None or self._charges is None:
            return np.inf
        return np.max(
            np.abs(self._charges - self._previous_charges)[: self._n_atoms]
        )

    @property
    def charges(self):
        if self._charges is None:
            return None
        return self._charges[self._array_indices]

    def _solve(self):
        self._previous_charges = copy.deepcopy(self._charges)
        try:
            self._charges = scipy.sparse.linalg.spsolve(
                self.coefficient_matrix, self.constant_vector
            )
        except RuntimeError as e:  # TODO: this could be slow?
            self._charges = scipy.sparse.linalg.lsmr(
                self.coefficient_matrix, self.constant_vector
            )[0]
        else:
            if np.isnan(self._charges[0]):
                self._charges = scipy.sparse.linalg.lsmr(
                    self.coefficient_matrix, self.constant_vector
                )[0]

    def _iter_solve(self, restraint_height, restraint_slope, b2):
        hyp_a = (restraint_height * self.n_structure_array)[self._array_indices]
        increment = hyp_a / np.sqrt(
            self._charges[self._array_indices] ** 2 + b2
        )
        self.coefficient_matrix = self._original_coefficient_matrix.copy()

        a_shape = self.coefficient_matrix[self._array_mask].shape
        self.coefficient_matrix[self._array_mask] += increment.reshape(a_shape)
        self._solve()
