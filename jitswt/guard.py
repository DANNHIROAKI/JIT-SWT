"""Guard management primitives used by the CPWL/JIT framework."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

from .linalg import Vector, as_vector, vector_norm


@dataclass(frozen=True)
class Halfspace:
    r"""Represents a single half-space :math:`a^\top x \le b`."""

    normal: Vector
    offset: float

    def as_tuple(self) -> Tuple[Vector, float]:
        return self.normal, self.offset


class GuardLibrary:
    """Stores unique half-spaces and exposes integer identifiers."""

    def __init__(self, dimension: int, atol: float = 1e-9) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        self._dim = dimension
        self._atol = atol
        self._guards: List[Halfspace] = []
        self._lookup: Dict[Tuple[int, ...], int] = {}

    @property
    def dimension(self) -> int:
        return self._dim

    def _normalise(self, normal: Iterable[float], offset: float) -> Tuple[Vector, float]:
        vector = as_vector(normal)
        if len(vector) != self._dim:
            raise ValueError("normal has wrong dimension")
        norm = vector_norm(vector)
        if norm <= self._atol:
            raise ValueError("normal vector must be non-zero")
        scale = 1.0 / norm
        normalised = tuple(scale * v for v in vector)
        return normalised, float(offset) * scale

    def _quantise_key(self, normal: Vector, offset: float) -> Tuple[int, ...]:
        factor = 1.0 / self._atol
        return tuple(int(round(v * factor)) for v in (*normal, offset))

    def register(self, normal: Iterable[float], offset: float) -> int:
        normalised, off = self._normalise(normal, offset)
        key = self._quantise_key(normalised, off)
        if key in self._lookup:
            return self._lookup[key]
        guard = Halfspace(normalised, off)
        idx = len(self._guards)
        self._guards.append(guard)
        self._lookup[key] = idx
        return idx

    def get(self, guard_id: int) -> Halfspace:
        try:
            return self._guards[guard_id]
        except IndexError as exc:  # pragma: no cover - defensive guard
            raise KeyError(f"unknown guard id {guard_id}") from exc

    def guards(self) -> Dict[int, Halfspace]:
        return {idx: guard for idx, guard in enumerate(self._guards)}


class GuardSet:
    r"""Finite set of guards representing :math:`\bigcap_i a_i^\top x \le b_i`."""

    def __init__(self, guard_ids: Iterable[int] = ()) -> None:
        self._ids = tuple(sorted(set(int(i) for i in guard_ids)))

    @property
    def ids(self) -> Tuple[int, ...]:
        return self._ids

    def add(self, guard_id: int) -> "GuardSet":
        return GuardSet(self._ids + (int(guard_id),))

    def union(self, other: "GuardSet") -> "GuardSet":
        return GuardSet(self._ids + other.ids)

    def __hash__(self) -> int:
        return hash(self._ids)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GuardSet):
            return NotImplemented
        return self._ids == other._ids

    def __repr__(self) -> str:
        return f"GuardSet(ids={self._ids})"
