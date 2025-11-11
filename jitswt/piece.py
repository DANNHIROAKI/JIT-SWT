"""Representation of linear pieces arising from CPWL networks."""

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
