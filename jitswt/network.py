"""Network compilation utilities for JIT-SWT."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Sequence

from .guard import GuardLibrary, GuardSet
from .linalg import Matrix, Vector, as_vector, eye, zeros
from .layers import AffineLayer, LeakyReLULayer, MaxLayer, ReLULayer
from .piece import LinearPiece
from .polytope import Polytope


def _as_matrix_from_rows(rows: Sequence[Sequence[float]]) -> Matrix:
    return tuple(tuple(float(x) for x in row) for row in rows)


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

    def add_max_pairs(self, pairs: Sequence[Sequence[int]]) -> None:
        self.layers.append(MaxLayer(pairs))

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
        identity = eye(self.input_dim)
        zeros_vec = zeros(self.input_dim)
        root_piece = LinearPiece(self.root_guard, self.domain, identity, zeros_vec)
        return self._propagate_piece(root_piece, 0, atol)

    def _propagate_piece(self, piece: LinearPiece, layer_idx: int, atol: float) -> List[LinearPiece]:
        if layer_idx >= len(self.layers):
            return [piece]
        layer = self.layers[layer_idx]
        if isinstance(layer, AffineLayer):
            next_piece = layer.apply(piece)
            return self._propagate_piece(next_piece, layer_idx + 1, atol)
        if isinstance(layer, ReLULayer):
            result: List[LinearPiece] = []
            for child in layer.apply(piece, self.library, atol):
                result.extend(self._propagate_piece(child, layer_idx + 1, atol))
            return result
        if isinstance(layer, LeakyReLULayer):
            result: List[LinearPiece] = []
            for child in layer.apply(piece, self.library, atol):
                result.extend(self._propagate_piece(child, layer_idx + 1, atol))
            return result
        if isinstance(layer, MaxLayer):
            result: List[LinearPiece] = []
            for child in layer.apply(piece, self.library, atol):
                result.extend(self._propagate_piece(child, layer_idx + 1, atol))
            return result
        raise TypeError(f"unsupported layer {layer}")

