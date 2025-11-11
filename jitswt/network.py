"""Network compilation utilities for JIT-SWT."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Sequence

from .guard import GuardLibrary, GuardSet
from .linalg import Matrix, Vector, as_vector, eye, zeros
from .layers import (
    AbsLayer,
    AffineLayer,
    LeakyReLULayer,
    MaxLayer,
    PReLULayer,
    ReLULayer,
)
from .piece import CompilationState, LinearPiece
from .polytope import Polytope


@dataclass
class NetworkBuilder:
    """Helper class to construct feed-forward CPWL networks."""

    input_dim: int
    layers: List[object] = field(default_factory=list)

    def add_affine(self, weight: Sequence[Sequence[float]], bias: Sequence[float]) -> None:
        self.layers.append(AffineLayer(weight, bias))

    def add_relu(self, size: int) -> None:
        self.layers.append(ReLULayer(size))

    def add_leaky_relu(self, size: int, alpha: float) -> None:
        self.layers.append(LeakyReLULayer(size, alpha))

    def add_prelu(self, slopes: Sequence[float]) -> None:
        self.layers.append(PReLULayer(slopes))

    def add_max_pairs(self, pairs: Sequence[Sequence[int]]) -> None:
        self.layers.append(MaxLayer(pairs))

    def add_abs(self, size: int) -> None:
        self.layers.append(AbsLayer(size))

    def build(self, domain: Polytope) -> "CPWLNetwork":
        return CPWLNetwork(self.input_dim, list(self.layers), domain)


@dataclass
class CPWLNetwork:
    input_dim: int
    layers: Sequence[object]
    domain: Polytope

    def __post_init__(self) -> None:
        self.library = GuardLibrary(self.input_dim)
        self.root_guard = GuardSet()
        for row, bound in zip(self.domain.A, self.domain.b):
            guard_id = self.library.register(row, bound)
            self.root_guard = self.root_guard.add(guard_id)

    def evaluate(self, x: Iterable[float]) -> Vector:
        vec = list(as_vector(x))
        if len(vec) != self.input_dim:
            raise ValueError("input dimension mismatch")
        current = vec
        for layer in self.layers:
            if isinstance(layer, AffineLayer):
                matrix = layer.weight
                bias = layer.bias
                current = [sum(matrix[i][k] * current[k] for k in range(len(current))) + bias[i] for i in range(len(matrix))]
            elif isinstance(layer, ReLULayer):
                current = [max(val, 0.0) for val in current]
            elif isinstance(layer, LeakyReLULayer):
                current = [val if val >= 0 else layer.alpha * val for val in current]
            elif isinstance(layer, MaxLayer):
                outputs = []
                for group in layer.groups:
                    if len(group) != 2:
                        raise NotImplementedError("MaxLayer evaluate currently supports pairs")
                    outputs.append(max(current[group[0]], current[group[1]]))
                current = outputs
            else:  # pragma: no cover - future extension
                raise TypeError(f"unsupported layer {layer}")
        return tuple(current)

    def enumerate_pieces(self, atol: float = 1e-9) -> List[LinearPiece]:
        stack: List[CompilationState] = [self.initial_state()]
        pieces: List[LinearPiece] = []
        while stack:
            state = stack.pop()
            if state.layer_idx >= len(self.layers):
                pieces.append(state.to_piece())
                continue
            layer = self.layers[state.layer_idx]
            if hasattr(layer, "propagate"):
                children = layer.propagate(state, self.library, atol)
                stack.extend(children)
            else:  # pragma: no cover - defensive guard
                raise TypeError(f"unsupported layer {layer}")
        return pieces

    def initial_state(self) -> CompilationState:
        identity = eye(self.input_dim)
        zeros_vec = zeros(self.input_dim)
        return CompilationState(self.root_guard, self.domain, identity, zeros_vec, layer_idx=0)

    def expand_state(self, state: CompilationState, atol: float = 1e-9) -> List[CompilationState]:
        if state.layer_idx >= len(self.layers):
            return []
        layer = self.layers[state.layer_idx]
        if hasattr(layer, "propagate"):
            return layer.propagate(state, self.library, atol)
        raise TypeError(f"unsupported layer {layer}")

