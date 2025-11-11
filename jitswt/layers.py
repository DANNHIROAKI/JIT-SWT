"""Layer definitions compatible with both static and JIT compilation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

from .guard import GuardLibrary
from .linalg import Matrix, Vector, add_vectors, matmul, matvec
from .piece import CompilationState, LinearPiece


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

    # ------------------------------------------------------------------
    # Static helpers ----------------------------------------------------
    # ------------------------------------------------------------------
    def apply(self, piece: LinearPiece) -> LinearPiece:
        matrix = matmul(self.weight, piece.matrix)
        bias = add_vectors(matvec(self.weight, piece.bias), self.bias)
        return LinearPiece(piece.guard, piece.polytope, matrix, bias)

    # ------------------------------------------------------------------
    # JIT propagation ---------------------------------------------------
    # ------------------------------------------------------------------
    def propagate(self, state: CompilationState, library: GuardLibrary, atol: float = 1e-9) -> List[CompilationState]:  # noqa: ARG002
        next_state = state.copy()
        next_state.matrix = matmul(self.weight, state.matrix)
        next_state.bias = add_vectors(matvec(self.weight, state.bias), self.bias)
        next_state.layer_idx = state.layer_idx + 1
        next_state.clear_pending()
        return [next_state]


@dataclass
class ReLULayer:
    size: int

    def apply(self, piece: LinearPiece, library: GuardLibrary, atol: float = 1e-9) -> List[LinearPiece]:
        states = [CompilationState(piece.guard, piece.polytope, piece.matrix, piece.bias, 0)]
        propagated = self.propagate(states[0], library, atol)
        return [state.to_piece() for state in propagated]

    def propagate(self, state: CompilationState, library: GuardLibrary, atol: float = 1e-9) -> List[CompilationState]:
        if state.output_dim() != self.size:
            raise ValueError("piece dimension mismatch")
        states = [state.copy()]
        for neuron in range(self.size):
            updated: List[CompilationState] = []
            for current in states:
                w = current.matrix[neuron]
                b = float(current.bias[neuron])
                lb, ub = current.polytope.bounds_on_linear_form(w, b, atol)
                if ub <= atol:
                    neg_state = current.copy()
                    zero_row = tuple(0.0 for _ in range(current.input_dim()))
                    rows = [list(row) for row in neg_state.matrix]
                    rows[neuron] = list(zero_row)
                    neg_state.matrix = tuple(tuple(row) for row in rows)
                    bias = list(neg_state.bias)
                    bias[neuron] = 0.0
                    neg_state.bias = tuple(bias)
                    updated.append(neg_state)
                elif lb >= -atol:
                    updated.append(current)
                else:
                    pos_state = current.copy()
                    neg_state = current.copy()
                    guard_pos = library.register([-v for v in w], b)
                    guard_neg = library.register(w, -b)
                    pos_state.guard = current.guard.add(guard_pos)
                    neg_state.guard = current.guard.add(guard_neg)
                    pos_state.polytope = current.polytope.intersection_with_halfspace([-v for v in w], b)
                    neg_state.polytope = current.polytope.intersection_with_halfspace(w, -b)
                    zero_row = tuple(0.0 for _ in range(current.input_dim()))
                    rows = [list(row) for row in neg_state.matrix]
                    rows[neuron] = list(zero_row)
                    neg_state.matrix = tuple(tuple(row) for row in rows)
                    bias = list(neg_state.bias)
                    bias[neuron] = 0.0
                    neg_state.bias = tuple(bias)
                    updated.extend([pos_state, neg_state])
            states = updated
        for s in states:
            s.layer_idx = state.layer_idx + 1
            s.clear_pending()
        return states


@dataclass
class LeakyReLULayer:
    size: int
    alpha: float

    def __post_init__(self) -> None:
        if not (0 <= self.alpha <= 1):
            raise ValueError("alpha must be in [0,1]")

    def apply(self, piece: LinearPiece, library: GuardLibrary, atol: float = 1e-9) -> List[LinearPiece]:
        states = [CompilationState(piece.guard, piece.polytope, piece.matrix, piece.bias, 0)]
        propagated = self.propagate(states[0], library, atol)
        return [state.to_piece() for state in propagated]

    def propagate(self, state: CompilationState, library: GuardLibrary, atol: float = 1e-9) -> List[CompilationState]:
        if state.output_dim() != self.size:
            raise ValueError("piece dimension mismatch")
        states = [state.copy()]
        for neuron in range(self.size):
            updated: List[CompilationState] = []
            for current in states:
                w = current.matrix[neuron]
                b = float(current.bias[neuron])
                lb, ub = current.polytope.bounds_on_linear_form(w, b, atol)
                if ub <= atol:
                    neg_state = current.copy()
                    rows = [list(row) for row in neg_state.matrix]
                    rows[neuron] = [self.alpha * v for v in rows[neuron]]
                    neg_state.matrix = tuple(tuple(row) for row in rows)
                    bias = list(neg_state.bias)
                    bias[neuron] *= self.alpha
                    neg_state.bias = tuple(bias)
                    updated.append(neg_state)
                elif lb >= -atol:
                    updated.append(current)
                else:
                    pos_state = current.copy()
                    neg_state = current.copy()
                    guard_pos = library.register([-v for v in w], b)
                    guard_neg = library.register(w, -b)
                    pos_state.guard = current.guard.add(guard_pos)
                    neg_state.guard = current.guard.add(guard_neg)
                    pos_state.polytope = current.polytope.intersection_with_halfspace([-v for v in w], b)
                    neg_state.polytope = current.polytope.intersection_with_halfspace(w, -b)
                    rows = [list(row) for row in neg_state.matrix]
                    rows[neuron] = [self.alpha * v for v in rows[neuron]]
                    neg_state.matrix = tuple(tuple(row) for row in rows)
                    bias = list(neg_state.bias)
                    bias[neuron] *= self.alpha
                    neg_state.bias = tuple(bias)
                    updated.extend([pos_state, neg_state])
            states = updated
        for s in states:
            s.layer_idx = state.layer_idx + 1
            s.clear_pending()
        return states


@dataclass
class PReLULayer:
    slopes: Sequence[float]

    def __post_init__(self) -> None:
        self.slopes = tuple(float(x) for x in self.slopes)
        if not self.slopes:
            raise ValueError("slopes must not be empty")
        if any(alpha < 0 for alpha in self.slopes):
            raise ValueError("PReLU slopes must be non-negative")

    def apply(self, piece: LinearPiece, library: GuardLibrary, atol: float = 1e-9) -> List[LinearPiece]:
        state = CompilationState(piece.guard, piece.polytope, piece.matrix, piece.bias, 0)
        propagated = self.propagate(state, library, atol)
        return [s.to_piece() for s in propagated]

    def propagate(self, state: CompilationState, library: GuardLibrary, atol: float = 1e-9) -> List[CompilationState]:
        if state.output_dim() != len(self.slopes):
            raise ValueError("piece dimension mismatch")
        states = [state.copy()]
        for neuron, alpha in enumerate(self.slopes):
            updated: List[CompilationState] = []
            for current in states:
                w = current.matrix[neuron]
                b = float(current.bias[neuron])
                lb, ub = current.polytope.bounds_on_linear_form(w, b, atol)
                if ub <= atol:
                    neg_state = current.copy()
                    rows = [list(row) for row in neg_state.matrix]
                    rows[neuron] = [alpha * v for v in rows[neuron]]
                    neg_state.matrix = tuple(tuple(row) for row in rows)
                    bias = list(neg_state.bias)
                    bias[neuron] *= alpha
                    neg_state.bias = tuple(bias)
                    updated.append(neg_state)
                elif lb >= -atol:
                    updated.append(current)
                else:
                    pos_state = current.copy()
                    neg_state = current.copy()
                    guard_pos = library.register([-v for v in w], b)
                    guard_neg = library.register(w, -b)
                    pos_state.guard = current.guard.add(guard_pos)
                    neg_state.guard = current.guard.add(guard_neg)
                    pos_state.polytope = current.polytope.intersection_with_halfspace([-v for v in w], b)
                    neg_state.polytope = current.polytope.intersection_with_halfspace(w, -b)
                    rows = [list(row) for row in neg_state.matrix]
                    rows[neuron] = [alpha * v for v in rows[neuron]]
                    neg_state.matrix = tuple(tuple(row) for row in rows)
                    bias = list(neg_state.bias)
                    bias[neuron] *= alpha
                    neg_state.bias = tuple(bias)
                    updated.extend([pos_state, neg_state])
            states = updated
        for s in states:
            s.layer_idx = state.layer_idx + 1
            s.clear_pending()
        return states


@dataclass
class MaxLayer:
    groups: Sequence[Sequence[int]]

    def apply(self, piece: LinearPiece, library: GuardLibrary, atol: float = 1e-9) -> List[LinearPiece]:
        state = CompilationState(piece.guard, piece.polytope, piece.matrix, piece.bias, 0)
        propagated = self.propagate(state, library, atol)
        return [s.to_piece() for s in propagated]

    def propagate(self, state: CompilationState, library: GuardLibrary, atol: float = 1e-9) -> List[CompilationState]:
        states = [state.copy()]
        for group in self.groups:
            if not group:
                raise ValueError("max group must be non-empty")
            if len(group) != 2:
                raise NotImplementedError("MaxLayer currently supports pairwise maxima only")
            updated: List[CompilationState] = []
            for current in states:
                i, j = group
                w_i = current.matrix[i]
                b_i = float(current.bias[i])
                w_j = current.matrix[j]
                b_j = float(current.bias[j])
                diff = tuple(w_i[k] - w_j[k] for k in range(len(w_i)))
                diff_bias = b_i - b_j
                lb, ub = current.polytope.bounds_on_linear_form(diff, diff_bias, atol)
                if lb >= -atol:
                    chosen = current.copy()
                    chosen.append_pending(w_i, b_i)
                    updated.append(chosen)
                elif ub <= atol:
                    chosen = current.copy()
                    chosen.append_pending(w_j, b_j)
                    updated.append(chosen)
                else:
                    pos_state = current.copy()
                    neg_state = current.copy()
                    guard_pos = library.register([-v for v in diff], diff_bias)
                    guard_neg = library.register(diff, -diff_bias)
                    pos_state.guard = current.guard.add(guard_pos)
                    neg_state.guard = current.guard.add(guard_neg)
                    pos_state.polytope = current.polytope.intersection_with_halfspace([-v for v in diff], diff_bias)
                    neg_state.polytope = current.polytope.intersection_with_halfspace(diff, -diff_bias)
                    pos_state.append_pending(w_i, b_i)
                    neg_state.append_pending(w_j, b_j)
                    updated.extend([pos_state, neg_state])
            states = updated
        final_states: List[CompilationState] = []
        for current in states:
            if len(current.pending_rows) != len(self.groups):
                raise RuntimeError("incomplete max assignments")
            finished = current.copy()
            finished.matrix = current.pending_rows
            finished.bias = current.pending_bias
            finished.layer_idx = state.layer_idx + 1
            finished.clear_pending()
            final_states.append(finished)
        return final_states


@dataclass
class AbsLayer:
    size: int

    def apply(self, piece: LinearPiece, library: GuardLibrary, atol: float = 1e-9) -> List[LinearPiece]:
        state = CompilationState(piece.guard, piece.polytope, piece.matrix, piece.bias, 0)
        propagated = self.propagate(state, library, atol)
        return [s.to_piece() for s in propagated]

    def propagate(self, state: CompilationState, library: GuardLibrary, atol: float = 1e-9) -> List[CompilationState]:
        if state.output_dim() != self.size:
            raise ValueError("piece dimension mismatch")
        states = [state.copy()]
        for neuron in range(self.size):
            updated: List[CompilationState] = []
            for current in states:
                w = current.matrix[neuron]
                b = float(current.bias[neuron])
                lb, ub = current.polytope.bounds_on_linear_form(w, b, atol)
                if ub <= atol:
                    neg_state = current.copy()
                    rows = [list(row) for row in neg_state.matrix]
                    rows[neuron] = [-v for v in rows[neuron]]
                    neg_state.matrix = tuple(tuple(row) for row in rows)
                    bias = list(neg_state.bias)
                    bias[neuron] = -bias[neuron]
                    neg_state.bias = tuple(bias)
                    updated.append(neg_state)
                elif lb >= -atol:
                    updated.append(current)
                else:
                    pos_state = current.copy()
                    neg_state = current.copy()
                    guard_pos = library.register([-v for v in w], b)
                    guard_neg = library.register(w, -b)
                    pos_state.guard = current.guard.add(guard_pos)
                    neg_state.guard = current.guard.add(guard_neg)
                    pos_state.polytope = current.polytope.intersection_with_halfspace([-v for v in w], b)
                    neg_state.polytope = current.polytope.intersection_with_halfspace(w, -b)
                    rows = [list(row) for row in neg_state.matrix]
                    rows[neuron] = [-v for v in rows[neuron]]
                    neg_state.matrix = tuple(tuple(row) for row in rows)
                    bias = list(neg_state.bias)
                    bias[neuron] = -bias[neuron]
                    neg_state.bias = tuple(bias)
                    updated.extend([pos_state, neg_state])
            states = updated
        for s in states:
            s.layer_idx = state.layer_idx + 1
            s.clear_pending()
        return states
