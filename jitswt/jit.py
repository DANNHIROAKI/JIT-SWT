"""Branch-and-bound analyzers built on top of the CPWL compilation."""
from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Tuple

import numpy as np

from .piece import LinearPiece
from .polytope import Polytope


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

    def maximize(self, coeff: Iterable[float], bias: float = 0.0) -> Tuple[float, LinearPiece]:
        coeff_arr = np.asarray(list(coeff), dtype=float)
        best_value = -np.inf
        best_piece: Optional[LinearPiece] = None
        for piece in self.pieces:
            linear_coeff = coeff_arr @ piece.matrix
            linear_bias = float(coeff_arr @ piece.bias + bias)
            lb, ub = piece.polytope.bounds_on_linear_form(linear_coeff, linear_bias)
            if ub > best_value:
                best_value = ub
                best_piece = piece
        if best_piece is None:
            raise RuntimeError("failed to compute maximum")
        return best_value, best_piece

    def evaluate_on_grid(self, points: Iterable[Iterable[float]]) -> List[Tuple[np.ndarray, np.ndarray]]:
        results = []
        for pt in points:
            x = np.asarray(list(pt), dtype=float)
            for piece in self.pieces:
                if piece.polytope.contains(x):
                    results.append((x, piece.evaluate(x)))
                    break
            else:  # pragma: no cover - unreachable for full cover
                raise ValueError("point outside all pieces")
        return results

    def piecewise_lipschitz(self, p_norm: float = 2.0) -> float:
        dual = p_norm / (p_norm - 1) if p_norm not in (np.inf, 1) else None
        if p_norm == np.inf:
            dual = 1
        elif p_norm == 1:
            dual = np.inf
        norms = []
        for piece in self.pieces:
            if dual == np.inf:
                row_norms = np.sum(np.abs(piece.matrix), axis=1)
            elif dual == 1:
                row_norms = np.max(np.abs(piece.matrix), axis=1)
            else:
                row_norms = np.linalg.norm(piece.matrix, ord=dual, axis=1)
            norms.append(np.max(row_norms))
        return float(np.max(norms))
