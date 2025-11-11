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
                    if pos_poly.is_feasible(atol):
                        pos_piece = current.copy()
                        pos_piece.guard = current.guard.add(pos_guard)
                        pos_piece.polytope = pos_poly
                        updated.append(pos_piece)
                    if neg_poly.is_feasible(atol):
                        neg_piece = current.copy()
                        neg_piece.guard = current.guard.add(neg_guard)
                        neg_piece.polytope = neg_poly
                        neg_piece.matrix[neuron] = 0.0
                        neg_piece.bias[neuron] = 0.0
                        updated.append(neg_piece)
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
                    if pos_poly.is_feasible(atol):
                        pos_piece = current.copy()
                        pos_piece.guard = current.guard.add(pos_guard)
                        pos_piece.polytope = pos_poly
                        updated.append(pos_piece)
                    if neg_poly.is_feasible(atol):
                        neg_piece = current.copy()
                        neg_piece.guard = current.guard.add(neg_guard)
                        neg_piece.polytope = neg_poly
                        neg_piece.matrix[neuron] *= self.alpha
                        neg_piece.bias[neuron] *= self.alpha
                        updated.append(neg_piece)
            pieces = updated
        return pieces


@dataclass
class MaxLayer:
    groups: Sequence[Sequence[int]]

    def apply(self, piece: LinearPiece, library: GuardLibrary, atol: float = 1e-9) -> List[LinearPiece]:
        selections: List[tuple[LinearPiece, List[int]]] = [(piece, [])]
        for group in self.groups:
            if not group:
                raise ValueError("max group must be non-empty")
            updated: List[tuple[LinearPiece, List[int]]] = []
            for current, chosen in selections:
                for idx in group:
                    candidate_piece = current.copy()
                    candidate_guard = current.guard
                    candidate_poly = current.polytope
                    feasible = True
                    for other in group:
                        if other == idx:
                            continue
                        w_i = candidate_piece.matrix[idx].copy()
                        b_i = float(candidate_piece.bias[idx])
                        w_j = candidate_piece.matrix[other].copy()
                        b_j = float(candidate_piece.bias[other])
                        lb_i, ub_i = candidate_poly.bounds_on_linear_form(w_i, b_i)
                        lb_j, ub_j = candidate_poly.bounds_on_linear_form(w_j, b_j)
                        if lb_i >= ub_j - atol:
                            continue
                        if lb_j >= ub_i - atol:
                            feasible = False
                            break
                        diff = w_i - w_j
                        offset = b_j - b_i
                        guard_normal = -diff
                        guard_offset = -offset
                        guard_id = library.register(guard_normal, guard_offset)
                        candidate_poly = candidate_poly.intersection_with_halfspace(guard_normal, guard_offset)
                        if not candidate_poly.is_feasible(atol):
                            feasible = False
                            break
                        candidate_guard = candidate_guard.add(guard_id)
                    if feasible and candidate_poly.is_feasible(atol):
                        candidate_piece.guard = candidate_guard
                        candidate_piece.polytope = candidate_poly
                        updated.append((candidate_piece, chosen + [idx]))
            selections = updated
        result: List[LinearPiece] = []
        for current, chosen in selections:
            indices = np.array(chosen, dtype=int)
            matrix = current.matrix[indices]
            bias = current.bias[indices]
            result.append(LinearPiece(current.guard, current.polytope, matrix, bias))
        return result
