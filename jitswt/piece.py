"""Representation of linear pieces arising from CPWL networks."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import List

import numpy as np

from .guard import GuardSet
from .polytope import Polytope


@dataclass
class LinearPiece:
    """Stores a region where the network behaves affinely."""

    guard: GuardSet
    polytope: Polytope
    matrix: np.ndarray
    bias: np.ndarray

    def copy(self) -> "LinearPiece":
        return LinearPiece(
            guard=self.guard,
            polytope=self.polytope,
            matrix=self.matrix.copy(),
            bias=self.bias.copy(),
        )

    def output_dim(self) -> int:
        return self.matrix.shape[0]

    def input_dim(self) -> int:
        return self.matrix.shape[1]

    def affine_parameters(self) -> List[tuple[np.ndarray, float]]:
        return [(self.matrix[i], float(self.bias[i])) for i in range(self.output_dim())]

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        return self.matrix @ x + self.bias
