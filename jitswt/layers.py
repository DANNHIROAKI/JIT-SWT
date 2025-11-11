"""Layer definitions for CPWL networks."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np

from .guard import GuardLibrary
from .piece import LinearPiece


@dataclass
class AffineLayer:
    weight: np.ndarray
    bias: np.ndarray

    def __post_init__(self) -> None:
        self.weight = np.asarray(self.weight, dtype=float)
        self.bias = np.asarray(self.bias, dtype=float)
        if self.bias.shape[0] != self.weight.shape[0]:
            raise ValueError("bias dimension mismatch")

    def apply(self, piece: LinearPiece) -> LinearPiece:
        matrix = self.weight @ piece.matrix
        bias = self.weight @ piece.bias + self.bias
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
                w = current.matrix[neuron].copy()
                b = float(current.bias[neuron])
                lb, ub = current.polytope.bounds_on_linear_form(w, b)
                if ub <= atol:
                    new_piece = current.copy()
                    new_piece.matrix[neuron] = 0.0
                    new_piece.bias[neuron] = 0.0
                    updated.append(new_piece)
                elif lb >= -atol:
                    updated.append(current)
                else:
                    pos_guard = library.register(-w, b)
                    neg_guard = library.register(w, -b)
                    pos_poly = current.polytope.intersection_with_halfspace(-w, b)
                    neg_poly = current.polytope.intersection_with_halfspace(w, -b)
                    pos_piece = current.copy()
                    pos_piece.guard = current.guard.add(pos_guard)
                    pos_piece.polytope = pos_poly
                    neg_piece = current.copy()
                    neg_piece.guard = current.guard.add(neg_guard)
                    neg_piece.polytope = neg_poly
                    neg_piece.matrix[neuron] = 0.0
                    neg_piece.bias[neuron] = 0.0
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
                w = current.matrix[neuron].copy()
                b = float(current.bias[neuron])
                lb, ub = current.polytope.bounds_on_linear_form(w, b)
                if ub <= atol:
                    new_piece = current.copy()
                    new_piece.matrix[neuron] *= self.alpha
                    new_piece.bias[neuron] *= self.alpha
                    updated.append(new_piece)
                elif lb >= -atol:
                    updated.append(current)
                else:
                    pos_guard = library.register(-w, b)
                    neg_guard = library.register(w, -b)
                    pos_poly = current.polytope.intersection_with_halfspace(-w, b)
                    neg_poly = current.polytope.intersection_with_halfspace(w, -b)
                    pos_piece = current.copy()
                    pos_piece.guard = current.guard.add(pos_guard)
                    pos_piece.polytope = pos_poly
                    neg_piece = current.copy()
                    neg_piece.guard = current.guard.add(neg_guard)
                    neg_piece.polytope = neg_poly
                    neg_piece.matrix[neuron] *= self.alpha
                    neg_piece.bias[neuron] *= self.alpha
                    updated.extend([pos_piece, neg_piece])
            pieces = updated
        return pieces


@dataclass
class MaxLayer:
    groups: Sequence[Sequence[int]]

    def apply(self, piece: LinearPiece, library: GuardLibrary, atol: float = 1e-9) -> List[LinearPiece]:
        if any(len(group) != 2 for group in self.groups):
            raise NotImplementedError("MaxLayer currently supports pairwise maxima only")
        pieces = [piece]
        for group_idx, group in enumerate(self.groups):
            if not group:
                raise ValueError("max group must be non-empty")
            updated: List[LinearPiece] = []
            for current in pieces:
                i, j = group
                w_i = current.matrix[i].copy()
                b_i = float(current.bias[i])
                w_j = current.matrix[j].copy()
                b_j = float(current.bias[j])
                lb_i, ub_i = current.polytope.bounds_on_linear_form(w_i, b_i)
                lb_j, ub_j = current.polytope.bounds_on_linear_form(w_j, b_j)
                if lb_i >= ub_j - atol:
                    new_piece = current.copy()
                    new_piece.matrix[group_idx] = w_i
                    new_piece.bias[group_idx] = b_i
                    updated.append(new_piece)
                elif lb_j >= ub_i - atol:
                    new_piece = current.copy()
                    new_piece.matrix[group_idx] = w_j
                    new_piece.bias[group_idx] = b_j
                    updated.append(new_piece)
                else:
                    diff = w_i - w_j
                    offset = b_j - b_i
                    pos_guard = library.register(-diff, -offset)
                    neg_guard = library.register(diff, offset)
                    pos_poly = current.polytope.intersection_with_halfspace(-diff, -offset)
                    neg_poly = current.polytope.intersection_with_halfspace(diff, offset)
                    pos_piece = current.copy()
                    pos_piece.guard = current.guard.add(pos_guard)
                    pos_piece.polytope = pos_poly
                    pos_piece.matrix[group_idx] = w_i
                    pos_piece.bias[group_idx] = b_i
                    neg_piece = current.copy()
                    neg_piece.guard = current.guard.add(neg_guard)
                    neg_piece.polytope = neg_poly
                    neg_piece.matrix[group_idx] = w_j
                    neg_piece.bias[group_idx] = b_j
                    updated.extend([pos_piece, neg_piece])
            pieces = updated
        return pieces
