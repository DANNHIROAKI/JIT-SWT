"""Representation of linear pieces and intermediate JIT states."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

from .guard import GuardSet
from .linalg import Matrix, Vector, as_vector, matvec
from .polytope import Polytope


@dataclass
class LinearPiece:
    """Stores a region where the network behaves affinely."""

    guard: GuardSet
    polytope: Polytope
    matrix: Matrix
    bias: Vector

    def copy(self) -> "LinearPiece":
        return LinearPiece(self.guard, self.polytope, tuple(tuple(row) for row in self.matrix), tuple(self.bias))

    def output_dim(self) -> int:
        return len(self.matrix)

    def input_dim(self) -> int:
        return len(self.matrix[0]) if self.matrix else 0

    def affine_parameters(self) -> List[Tuple[Vector, float]]:
        return [(row, float(self.bias[i])) for i, row in enumerate(self.matrix)]

    def evaluate(self, x: Sequence[float]) -> Vector:
        return tuple(val + self.bias[i] for i, val in enumerate(matvec(self.matrix, as_vector(x))))


@dataclass
class CompilationState:
    """Represents a partially compiled region of the network.

    The state tracks how far the symbolic compilation progressed (``layer_idx``)
    together with the affine representation of the currently materialised
    sub-network.  When ``layer_idx`` equals the number of layers in the network
    the state can be converted directly into a :class:`LinearPiece` via
    :meth:`to_piece`.
    """

    guard: GuardSet
    polytope: Polytope
    matrix: Matrix
    bias: Vector
    layer_idx: int
    pending_rows: Tuple[Vector, ...] = ()
    pending_bias: Tuple[float, ...] = ()

    def copy(self) -> "CompilationState":
        return CompilationState(
            self.guard,
            self.polytope,
            tuple(tuple(row) for row in self.matrix),
            tuple(self.bias),
            self.layer_idx,
            tuple(self.pending_rows),
            tuple(self.pending_bias),
        )

    def output_dim(self) -> int:
        return len(self.matrix)

    def input_dim(self) -> int:
        return len(self.matrix[0]) if self.matrix else 0

    def append_pending(self, row: Vector, bias: float) -> None:
        rows = list(self.pending_rows)
        rows.append(tuple(row))
        self.pending_rows = tuple(rows)
        biases = list(self.pending_bias)
        biases.append(float(bias))
        self.pending_bias = tuple(biases)

    def clear_pending(self) -> None:
        self.pending_rows = ()
        self.pending_bias = ()

    def to_piece(self) -> LinearPiece:
        return LinearPiece(self.guard, self.polytope, self.matrix, self.bias)
