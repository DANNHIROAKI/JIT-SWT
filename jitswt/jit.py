"""Branch-and-bound analyzers built on top of the CPWL compilation."""


from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

from .linalg import Vector, as_vector, matrix_row_norm
from .network import CPWLNetwork
from .piece import CompilationState, LinearPiece


@dataclass
class MaximizationResult:
    """Result container returned by maximization queries."""

    upper_bound: float
    lower_bound: float
    certificate: Optional[LinearPiece]
    status: str
    explored_pieces: int


class BranchAndBoundAnalyzer:
    """Runs branch-and-bound queries over a pre-enumerated piece list."""

    def __init__(self, pieces: Iterable[LinearPiece]):
        self.pieces = list(pieces)
        if not self.pieces:
            raise ValueError("no pieces provided")

    def maximize(self, coeff: Sequence[float], bias: float = 0.0) -> MaximizationResult:
        coeff_vec = as_vector(coeff)
        best_upper = float("-inf")
        best_lower = float("-inf")
        best_upper_piece: Optional[LinearPiece] = None
        best_lower_piece: Optional[LinearPiece] = None
        for piece in self.pieces:
            w, linear_bias = _objective_parameters(piece, coeff_vec, bias)
            lb, ub = piece.polytope.bounds_on_linear_form(w, linear_bias)
            if ub > best_upper:
                best_upper = ub
                best_upper_piece = piece
            if lb > best_lower:
                best_lower = lb
                best_lower_piece = piece
        certificate = best_lower_piece or best_upper_piece
        if certificate is None:  # pragma: no cover - defensive guard
            raise RuntimeError("failed to compute maximum")
        return MaximizationResult(
            upper_bound=best_upper,
            lower_bound=best_lower,
            certificate=certificate,
            status="complete",
            explored_pieces=len(self.pieces),
        )

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


class JITBranchAndBound:
    """Branch-and-bound analyser that refines network pieces on demand."""

    def __init__(self, network: CPWLNetwork, atol: float = 1e-9) -> None:
        self.network = network
        self.atol = atol

    def maximize(
        self,
        coeff: Sequence[float],
        bias: float = 0.0,
        *,
        budget: Optional[int] = None,
    ) -> MaximizationResult:
        coeff_vec = as_vector(coeff)
        pending: List[CompilationState] = [self.network.initial_state()]
        pieces_explored = 0
        refinements = 0
        best_upper = float("-inf")
        best_lower = float("-inf")
        best_upper_piece: Optional[LinearPiece] = None
        best_lower_piece: Optional[LinearPiece] = None
        status = "complete"

        while pending:
            state = pending.pop()
            if state.layer_idx < len(self.network.layers):
                if budget is not None and refinements >= budget:
                    pending.append(state)
                    status = "budget"
                    break
                children = self.network.expand_state(state, self.atol)
                pending.extend(children)
                refinements += 1
                continue
            piece = state.to_piece()
            w, linear_bias = _objective_parameters(piece, coeff_vec, bias)
            lb, ub = piece.polytope.bounds_on_linear_form(w, linear_bias)
            pieces_explored += 1
            if ub > best_upper:
                best_upper = ub
                best_upper_piece = piece
            if lb > best_lower:
                best_lower = lb
                best_lower_piece = piece

        certificate = best_lower_piece or best_upper_piece
        if certificate is None:
            if status == "budget":
                upper = best_upper if best_upper != float("-inf") else float("-inf")
                lower = best_lower if best_lower != float("-inf") else upper
                return MaximizationResult(
                    upper_bound=upper,
                    lower_bound=lower,
                    certificate=None,
                    status=status,
                    explored_pieces=pieces_explored,
                )
            raise RuntimeError("no feasible pieces discovered")
        lower = best_lower if best_lower != float("-inf") else best_upper
        upper = best_upper if best_upper != float("-inf") else lower

        return MaximizationResult(
            upper_bound=upper,
            lower_bound=lower,
            certificate=certificate,
            status=status,
            explored_pieces=pieces_explored,
        )


def _objective_parameters(piece: LinearPiece, coeff_vec: Vector, bias: float) -> Tuple[List[float], float]:
    input_dim = piece.input_dim()
    w = [0.0] * input_dim
    for row_idx, coeff_val in enumerate(coeff_vec):
        row = piece.matrix[row_idx]
        for col_idx in range(input_dim):
            w[col_idx] += coeff_val * row[col_idx]
    linear_bias = sum(coeff_vec[i] * piece.bias[i] for i in range(len(coeff_vec))) + bias
    return w, linear_bias

