"""Network compilation utilities for JIT-SWT."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Sequence

import numpy as np

from .guard import GuardLibrary, GuardSet
from .layers import AffineLayer, LeakyReLULayer, MaxLayer, ReLULayer
from .piece import LinearPiece
from .polytope import Polytope


@dataclass
class NetworkBuilder:
    """Helper class to construct feed-forward CPWL networks."""

    input_dim: int
    layers: List[object] = field(default_factory=list)

    def add_affine(self, weight: np.ndarray, bias: np.ndarray) -> None:
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

    def evaluate(self, x: Iterable[float]) -> np.ndarray:
        vec = np.asarray(list(x), dtype=float)
        if vec.shape != (self.input_dim,):
            raise ValueError("input dimension mismatch")
        current = vec
        for layer in self.layers:
            if isinstance(layer, AffineLayer):
                current = layer.weight @ current + layer.bias
            elif isinstance(layer, ReLULayer):
                current = np.maximum(current, 0.0)
            elif isinstance(layer, LeakyReLULayer):
                current = np.where(current >= 0, current, layer.alpha * current)
            elif isinstance(layer, MaxLayer):
                outputs = []
                for group in layer.groups:
                    if not group:
                        raise ValueError("max group must contain at least one index")
                    outputs.append(np.max(current[np.array(group, dtype=int)]))
                current = np.asarray(outputs)
            else:  # pragma: no cover - future extension
                raise TypeError(f"unsupported layer {layer}")
        return current

    def enumerate_pieces(self, atol: float = 1e-9) -> List[LinearPiece]:
        identity = np.eye(self.input_dim)
        zeros = np.zeros(self.input_dim)
        root_piece = LinearPiece(self.root_guard, self.domain, identity, zeros)
        pieces = self._propagate_piece(root_piece, 0, atol)
        return pieces

    def _propagate_piece(self, piece: LinearPiece, layer_idx: int, atol: float) -> List[LinearPiece]:
        if layer_idx >= len(self.layers):
            return [piece]
        layer = self.layers[layer_idx]
        if isinstance(layer, AffineLayer):
            next_piece = layer.apply(piece)
            return self._propagate_piece(next_piece, layer_idx + 1, atol)
        if isinstance(layer, ReLULayer):
            pieces = layer.apply(piece, self.library, atol)
            result: List[LinearPiece] = []
            for child in pieces:
                result.extend(self._propagate_piece(child, layer_idx + 1, atol))
            return result
        if isinstance(layer, LeakyReLULayer):
            pieces = layer.apply(piece, self.library, atol)
            result: List[LinearPiece] = []
            for child in pieces:
                result.extend(self._propagate_piece(child, layer_idx + 1, atol))
            return result
        if isinstance(layer, MaxLayer):
            pieces = layer.apply(piece, self.library, atol)
            result: List[LinearPiece] = []
            for child in pieces:
                result.extend(self._propagate_piece(child, layer_idx + 1, atol))
            return result
        raise TypeError(f"unsupported layer {layer}")
