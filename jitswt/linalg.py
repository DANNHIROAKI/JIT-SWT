"""Lightweight linear algebra helpers used throughout the project.

The initial prototype of the repository relied on :mod:`numpy` for even the
most basic linear algebra manipulations.  That made the code impossible to run
in environments where compiled extensions are unavailable (as witnessed by the
previous CI failure).  The implementation in this module purposely keeps the
surface area extremely small while still providing the operations required by
the CPWL/JIT tooling: matrix multiplication, solving small dense linear
systems, norms and a numerically robust rank computation.

The helpers operate purely on Python ``list`` and ``tuple`` containers filled
with ``float`` values.  They remain compatible with ``numpy`` arrays – the
``as_vector``/``as_matrix`` functions coerce any iterable of numbers to the
internal representation.  Keeping the representation explicit removes the
dependency on ``numpy`` while still offering deterministic behaviour that is
straightforward to unit test.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import zip_longest
from typing import Iterable, List, Sequence, Tuple


Vector = Tuple[float, ...]
Matrix = Tuple[Tuple[float, ...], ...]


def as_vector(values: Iterable[float]) -> Vector:
    """Return a tuple containing ``values`` cast to ``float``.

    Using tuples avoids accidental aliasing when vectors are shared between
    multiple objects.  The function accepts any iterable so callers can pass in
    generators or lists without a separate conversion step.
    """

    return tuple(float(v) for v in values)


def as_matrix(rows: Iterable[Iterable[float]]) -> Matrix:
    """Convert *rows* into the canonical matrix representation.

    The function validates that all rows have the same width.  A ``ValueError``
    is raised when an empty matrix is provided or when the row lengths are
    inconsistent; the guard surfaces bugs early instead of letting downstream
    computations fail in hard-to-debug locations.
    """

    converted: List[Vector] = [as_vector(row) for row in rows]
    if not converted:
        raise ValueError("matrix must contain at least one row")
    width = len(converted[0])
    for row in converted:
        if len(row) != width:
            raise ValueError("inconsistent row width")
    return tuple(converted)


def zeros(length: int) -> Vector:
    return tuple(0.0 for _ in range(length))


def eye(size: int) -> Matrix:
    rows: List[Tuple[float, ...]] = []
    for i in range(size):
        row = [0.0] * size
        row[i] = 1.0
        rows.append(tuple(row))
    return tuple(rows)


def add_vectors(lhs: Vector, rhs: Vector) -> Vector:
    return tuple(a + b for a, b in zip(lhs, rhs))


def subtract_vectors(lhs: Vector, rhs: Vector) -> Vector:
    return tuple(a - b for a, b in zip(lhs, rhs))


def scale_vector(vec: Vector, scalar: float) -> Vector:
    return tuple(scalar * v for v in vec)


def dot(lhs: Vector, rhs: Vector) -> float:
    return sum(a * b for a, b in zip(lhs, rhs))


def matvec(matrix: Matrix, vector: Vector) -> Vector:
    if not matrix:
        raise ValueError("matrix must not be empty")
    if len(matrix[0]) != len(vector):
        raise ValueError("dimension mismatch in matvec")
    return tuple(dot(row, vector) for row in matrix)


def matmul(lhs: Matrix, rhs: Matrix) -> Matrix:
    if len(lhs[0]) != len(rhs):
        raise ValueError("dimension mismatch in matmul")
    rhs_t = transpose(rhs)
    result_rows: List[Tuple[float, ...]] = []
    for row in lhs:
        result_rows.append(tuple(dot(row, col) for col in rhs_t))
    return tuple(result_rows)


def transpose(matrix: Matrix) -> Matrix:
    height = len(matrix)
    width = len(matrix[0])
    cols: List[List[float]] = [[0.0] * height for _ in range(width)]
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            cols[j][i] = value
    return tuple(tuple(col) for col in cols)


def vector_norm(vec: Vector, order: float = 2.0) -> float:
    if order == 1:
        return sum(abs(v) for v in vec)
    if order == 2:
        return sum(v * v for v in vec) ** 0.5
    if order == float("inf"):
        return max(abs(v) for v in vec)
    return sum(abs(v) ** order for v in vec) ** (1.0 / order)


def matrix_row_norm(row: Vector, order: float) -> float:
    if order == float("inf"):
        return max(abs(v) for v in row)
    if order == 1:
        return sum(abs(v) for v in row)
    return vector_norm(row, order)


def matrix_rank(matrix: Matrix, atol: float = 1e-9) -> int:
    """Return the numerical rank using Gaussian elimination."""

    m = [list(row) for row in matrix]
    height = len(m)
    width = len(m[0])
    rank = 0
    col = 0
    for row in range(height):
        pivot = None
        for r in range(row, height):
            if abs(m[r][col]) > atol:
                pivot = r
                break
        while pivot is None:
            col += 1
            if col >= width:
                return rank
            for r in range(row, height):
                if abs(m[r][col]) > atol:
                    pivot = r
                    break
        if pivot != row:
            m[row], m[pivot] = m[pivot], m[row]
        factor = m[row][col]
        rank += 1
        for c in range(col, width):
            m[row][c] /= factor
        for r in range(height):
            if r == row:
                continue
            factor = m[r][col]
            if abs(factor) <= atol:
                continue
            for c in range(col, width):
                m[r][c] -= factor * m[row][c]
        col += 1
        if col >= width:
            break
    return rank


def solve(matrix: Matrix, rhs: Vector, atol: float = 1e-9) -> Vector:
    """Solve ``matrix * x = rhs`` using Gaussian elimination.

    The function mirrors :func:`numpy.linalg.solve` for the small dense systems
    encountered when enumerating vertices of a polytope.  A ``ValueError`` is
    raised when the system is singular within the provided tolerance.
    """

    m = [list(row) for row in matrix]
    b = list(rhs)
    n = len(m)
    if len(m[0]) != n or len(rhs) != n:
        raise ValueError("matrix must be square and match rhs dimensions")
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[pivot][col]) <= atol:
            raise ValueError("matrix is singular")
        if pivot != col:
            m[col], m[pivot] = m[pivot], m[col]
            b[col], b[pivot] = b[pivot], b[col]
        pivot_val = m[col][col]
        for c in range(col, n):
            m[col][c] /= pivot_val
        b[col] /= pivot_val
        for r in range(n):
            if r == col:
                continue
            factor = m[r][col]
            if abs(factor) <= atol:
                continue
            for c in range(col, n):
                m[r][c] -= factor * m[col][c]
            b[r] -= factor * b[col]
    return tuple(b)


def is_close(lhs: float, rhs: float, atol: float = 1e-9) -> bool:
    return abs(lhs - rhs) <= atol


def almost_equal_vectors(lhs: Vector, rhs: Vector, atol: float = 1e-9) -> bool:
    return all(is_close(a, b, atol) for a, b in zip_longest(lhs, rhs, fillvalue=0.0))


@dataclass(frozen=True)
class LinearFunctional:
    """Represents ``w^T x + b``; handy container for propagation code."""

    coeff: Vector
    bias: float

    def evaluate(self, vector: Vector) -> float:
        return dot(self.coeff, vector) + self.bias

