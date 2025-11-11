"""Guard library and guard sets used by symbolic weighted transducers.

The implementation follows the definitions in the paper.  Guards are stored as
half-spaces :math:`a^\top x \le b` and are normalised to enforce uniqueness.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np


@dataclass(frozen=True)
class Halfspace:
    """Represents a single half-space :math:`a^\top x \le b`."""

    normal: np.ndarray
    offset: float

    def as_tuple(self) -> Tuple[Tuple[float, ...], float]:
        return tuple(np.asarray(self.normal).tolist()), float(self.offset)


class GuardLibrary:
    """Stores unique half-spaces and exposes integer identifiers."""

    def __init__(self, dimension: int, atol: float = 1e-9) -> None:
        self._dim = dimension
        self._atol = atol
        self._guards: Dict[Tuple[int, ...], Halfspace] = {}
        self._lookup: Dict[Tuple[int, ...], int] = {}
        self._by_id: Dict[int, Halfspace] = {}
        self._next_id = 0

    @property
    def dimension(self) -> int:
        return self._dim

    def register(self, normal: Iterable[float], offset: float) -> int:
        normal_arr = np.asarray(list(normal), dtype=float)
        if normal_arr.shape != (self._dim,):
            raise ValueError("normal has wrong dimension")
        norm = np.linalg.norm(normal_arr)
        if norm <= self._atol:
            raise ValueError("normal vector must be non-zero")
        normal_arr = normal_arr / norm
        offset = float(offset) / norm
        key = tuple(int(np.round(v / self._atol)) for v in np.concatenate([normal_arr, [offset]]))
        if key in self._lookup:
            return self._lookup[key]
        guard = Halfspace(normal=normal_arr, offset=offset)
        self._guards[key] = guard
        idx = self._next_id
        self._lookup[key] = idx
        self._by_id[idx] = guard
        self._next_id += 1
        return idx

    def get(self, guard_id: int) -> Halfspace:
        try:
            return self._by_id[guard_id]
        except KeyError as exc:  # pragma: no cover - defensive programming
            raise KeyError(f"unknown guard id {guard_id}") from exc

    def guards(self) -> Dict[int, Halfspace]:
        return dict(self._by_id)


class GuardSet:
    """Finite set of guards representing :math:`\bigcap_i a_i^\top x \le b_i`."""

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
