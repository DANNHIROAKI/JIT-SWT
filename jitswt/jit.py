"""Branch-and-bound analyzers built on top of the CPWL compilation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

from .linalg import Vector, as_vector, matrix_row_norm
from .piece import LinearPiece


@dataclass(order=True)
class _QueueEntry:
    priority: float
    upper: float
    lower: float
    piece: LinearPiece


class BranchAndBoundAnalyzer:
    """Runs branch-and-bound queries over a compiled network."""

    def __init__(self, pieces: Iterable[LinearPiece]):
        self.pieces = list(pieces)
        if not self.pieces:
            raise ValueError("no pieces provided")

    def maximize(self, coeff: Sequence[float], bias: float = 0.0) -> Tuple[float, LinearPiece]:
        coeff_vec = as_vector(coeff)
        best_value = float("-inf")
        best_piece: Optional[LinearPiece] = None
        for piece in self.pieces:
            w = [0.0] * piece.input_dim()
            for row_idx, coeff_val in enumerate(coeff_vec):
                for col_idx in range(piece.input_dim()):
                    w[col_idx] += coeff_val * piece.matrix[row_idx][col_idx]
            linear_bias = sum(coeff_vec[i] * piece.bias[i] for i in range(len(coeff_vec))) + bias
            lb, ub = piece.polytope.bounds_on_linear_form(w, linear_bias)
            if ub > best_value:
                best_value = ub
                best_piece = piece
        if best_piece is None:  # pragma: no cover - defensive guard
            raise RuntimeError("failed to compute maximum")
        return best_value, best_piece

    def evaluate_on_grid(self, points: Iterable[Sequence[float]]) -> List[Tuple[Vector, Vector]]:
        results = []
        for pt in points:
            x = as_vector(pt)
            for piece in self.pieces:
                if piece.polytope.contains(x):
                    results.append((x, piece.evaluate(x)))
                    break
            else:  # pragma: no cover - unreachable for full cover
                raise ValueError("point outside all pieces")
        return results

    def piecewise_lipschitz(self, p_norm: float = 2.0) -> float:
        if p_norm == 1:
            dual = float("inf")
        elif p_norm == float("inf"):
            dual = 1.0
        else:
            dual = p_norm / (p_norm - 1)
        bounds = []
        for piece in self.pieces:
            row_norms = [matrix_row_norm(row, dual) for row in piece.matrix]
            bounds.append(max(row_norms))
        return max(bounds)

