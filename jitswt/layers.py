"""Layer definitions for CPWL networks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

from .guard import GuardLibrary
from .linalg import Matrix, Vector, add_vectors, matmul, matvec
from .piece import LinearPiece


def _to_matrix(value: Iterable[Iterable[float]]) -> Matrix:
    return tuple(tuple(float(x) for x in row) for row in value)


def _to_vector(value: Iterable[float]) -> Vector:
    return tuple(float(x) for x in value)


@dataclass
class AffineLayer:
    weight: Matrix
    bias: Vector

    def __post_init__(self) -> None:
        self.weight = _to_matrix(self.weight)
        self.bias = _to_vector(self.bias)
        if len(self.bias) != len(self.weight):
            raise ValueError("bias dimension mismatch")

    def apply(self, piece: LinearPiece) -> LinearPiece:
        matrix = matmul(self.weight, piece.matrix)
        bias = add_vectors(matvec(self.weight, piece.bias), self.bias)
        return LinearPiece(piece.guard, piece.polytope, matrix, bias)


@dataclass
class ReLULayer:
    size: int

    def apply(self, piece: LinearPiece, library: GuardLibrary, atol: float = 1e-9) -> List[LinearPiece]:
        if piece.output_dim() != self.size:
            raise ValueError("piece dimension mismatch")
        pieces = [piece]
        for neuron in range(self.size):
            updated: List[LinearPiece] = []
            for current in pieces:
                w = current.matrix[neuron]
                b = float(current.bias[neuron])
                lb, ub = current.polytope.bounds_on_linear_form(w, b, atol)
                if ub <= atol:
                    new_piece = current.copy()
                    zero_row = tuple(0.0 for _ in range(current.input_dim()))
                    new_bias = list(new_piece.bias)
                    new_bias[neuron] = 0.0
                    new_rows = [list(row) for row in new_piece.matrix]
                    new_rows[neuron] = list(zero_row)
                    new_piece.matrix = tuple(tuple(row) for row in new_rows)
                    new_piece.bias = tuple(new_bias)
                    updated.append(new_piece)
                elif lb >= -atol:
                    updated.append(current)
                else:
                    pos_guard = library.register([-v for v in w], b)
                    neg_guard = library.register(w, -b)
                    pos_poly = current.polytope.intersection_with_halfspace([-v for v in w], b)
                    neg_poly = current.polytope.intersection_with_halfspace(w, -b)
                    pos_piece = current.copy()
                    pos_piece.guard = current.guard.add(pos_guard)
                    pos_piece.polytope = pos_poly
                    neg_piece = current.copy()
                    neg_piece.guard = current.guard.add(neg_guard)
                    neg_piece.polytope = neg_poly
                    zero_row = tuple(0.0 for _ in range(current.input_dim()))
                    neg_rows = [list(row) for row in neg_piece.matrix]
                    neg_rows[neuron] = list(zero_row)
                    neg_piece.matrix = tuple(tuple(row) for row in neg_rows)
                    neg_bias = list(neg_piece.bias)
                    neg_bias[neuron] = 0.0
                    neg_piece.bias = tuple(neg_bias)
                    updated.extend([pos_piece, neg_piece])
            pieces = updated
        return pieces


@dataclass
class LeakyReLULayer:
    size: int
    alpha: float

    def apply(self, piece: LinearPiece, library: GuardLibrary, atol: float = 1e-9) -> List[LinearPiece]:
        if piece.output_dim() != self.size:
            raise ValueError("piece dimension mismatch")
        if not (0 <= self.alpha <= 1):
            raise ValueError("alpha must be in [0,1]")
        pieces = [piece]
        for neuron in range(self.size):
            updated: List[LinearPiece] = []
            for current in pieces:
                w = current.matrix[neuron]
                b = float(current.bias[neuron])
                lb, ub = current.polytope.bounds_on_linear_form(w, b, atol)
                if ub <= atol:
                    new_piece = current.copy()
                    new_rows = [list(row) for row in new_piece.matrix]
                    new_rows[neuron] = [self.alpha * v for v in new_rows[neuron]]
                    new_piece.matrix = tuple(tuple(row) for row in new_rows)
                    new_bias = list(new_piece.bias)
                    new_bias[neuron] *= self.alpha
                    new_piece.bias = tuple(new_bias)
                    updated.append(new_piece)
                elif lb >= -atol:
                    updated.append(current)
                else:
                    pos_guard = library.register([-v for v in w], b)
                    neg_guard = library.register(w, -b)
                    pos_poly = current.polytope.intersection_with_halfspace([-v for v in w], b)
                    neg_poly = current.polytope.intersection_with_halfspace(w, -b)
                    pos_piece = current.copy()
                    pos_piece.guard = current.guard.add(pos_guard)
                    pos_piece.polytope = pos_poly
                    neg_piece = current.copy()
                    neg_piece.guard = current.guard.add(neg_guard)
                    neg_piece.polytope = neg_poly
                    neg_rows = [list(row) for row in neg_piece.matrix]
                    neg_rows[neuron] = [self.alpha * v for v in neg_rows[neuron]]
                    neg_piece.matrix = tuple(tuple(row) for row in neg_rows)
                    neg_bias = list(neg_piece.bias)
                    neg_bias[neuron] *= self.alpha
                    neg_piece.bias = tuple(neg_bias)
                    updated.extend([pos_piece, neg_piece])
            pieces = updated
        return pieces


@dataclass
class MaxLayer:
    groups: Sequence[Sequence[int]]

    def apply(self, piece: LinearPiece, library: GuardLibrary, atol: float = 1e-9) -> List[LinearPiece]:
        pieces = [piece]
        for group_idx, group in enumerate(self.groups):
            if not group:
                raise ValueError("max group must be non-empty")
            if len(group) != 2:
                raise NotImplementedError("MaxLayer currently supports pairwise maxima only")
            updated: List[LinearPiece] = []
            for current in pieces:
                i, j = group
                w_i = current.matrix[i]
                b_i = float(current.bias[i])
                w_j = current.matrix[j]
                b_j = float(current.bias[j])
                lb_i, ub_i = current.polytope.bounds_on_linear_form(w_i, b_i, atol)
                lb_j, ub_j = current.polytope.bounds_on_linear_form(w_j, b_j, atol)
                if lb_i >= ub_j - atol:
                    new_piece = current.copy()
                    new_rows = [list(row) for row in new_piece.matrix]
                    new_rows[group_idx] = list(w_i)
                    new_piece.matrix = tuple(tuple(row) for row in new_rows)
                    new_bias = list(new_piece.bias)
                    new_bias[group_idx] = b_i
                    new_piece.bias = tuple(new_bias)
                    updated.append(new_piece)
                elif lb_j >= ub_i - atol:
                    new_piece = current.copy()
                    new_rows = [list(row) for row in new_piece.matrix]
                    new_rows[group_idx] = list(w_j)
                    new_piece.matrix = tuple(tuple(row) for row in new_rows)
                    new_bias = list(new_piece.bias)
                    new_bias[group_idx] = b_j
                    new_piece.bias = tuple(new_bias)
                    updated.append(new_piece)
                else:
                    diff = tuple(w_i[k] - w_j[k] for k in range(len(w_i)))
                    offset = b_j - b_i
                    pos_guard = library.register([-v for v in diff], -offset)
                    neg_guard = library.register(diff, offset)
                    pos_poly = current.polytope.intersection_with_halfspace([-v for v in diff], -offset)
                    neg_poly = current.polytope.intersection_with_halfspace(diff, offset)
                    pos_piece = current.copy()
                    pos_piece.guard = current.guard.add(pos_guard)
                    pos_piece.polytope = pos_poly
                    pos_rows = [list(row) for row in pos_piece.matrix]
                    pos_rows[group_idx] = list(w_i)
                    pos_piece.matrix = tuple(tuple(row) for row in pos_rows)
                    pos_bias = list(pos_piece.bias)
                    pos_bias[group_idx] = b_i
                    pos_piece.bias = tuple(pos_bias)
                    neg_piece = current.copy()
                    neg_piece.guard = current.guard.add(neg_guard)
                    neg_piece.polytope = neg_poly
                    neg_rows = [list(row) for row in neg_piece.matrix]
                    neg_rows[group_idx] = list(w_j)
                    neg_piece.matrix = tuple(tuple(row) for row in neg_rows)
                    neg_bias = list(neg_piece.bias)
                    neg_bias[group_idx] = b_j
                    neg_piece.bias = tuple(neg_bias)
                    updated.extend([pos_piece, neg_piece])
            pieces = updated
        return pieces

