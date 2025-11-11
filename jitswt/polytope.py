"""Basic polytope utilities used by the CPWL/JIT toolchain."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, List, Sequence, Tuple

from .linalg import Matrix, Vector, as_vector, matrix_rank, matvec, solve


@dataclass
class Polytope:
    r"""Convex polytope :math:`\{x \mid Ax \le b\}`."""

    A: Matrix
    b: Vector

    def __post_init__(self) -> None:
        if len(self.A) != len(self.b):
            raise ValueError("A and b shapes incompatible")
        if not self.A:
            raise ValueError("matrix A must not be empty")
        width = len(self.A[0])
        for row in self.A:
            if len(row) != width:
                raise ValueError("matrix rows must have equal length")
        self._dim = width

    @property
    def dimension(self) -> int:
        return self._dim

    def intersection_with_halfspace(self, normal: Iterable[float], offset: float) -> "Polytope":
        normal_vec = as_vector(normal)
        if len(normal_vec) != self.dimension:
            raise ValueError("dimension mismatch")
        new_A = tuple(self.A + (normal_vec,))
        new_b = tuple(self.b + (float(offset),))
        return Polytope(new_A, new_b)

    def contains(self, point: Sequence[float], atol: float = 1e-9) -> bool:
        x = as_vector(point)
        if len(x) != self.dimension:
            raise ValueError("dimension mismatch")
        values = matvec(self.A, x)
        return all(val <= bound + atol for val, bound in zip(values, self.b))

    def bounds_on_linear_form(self, coeff: Iterable[float], bias: float = 0.0, atol: float = 1e-9) -> Tuple[float, float]:
        coeff_vec = as_vector(coeff)
        if len(coeff_vec) != self.dimension:
            raise ValueError("dimension mismatch")
        bias = float(bias)
        vertices = self._enumerate_vertices(atol)
        if not vertices:
            raise RuntimeError("polytope appears empty or unbounded")
        values = [sum(c * x for c, x in zip(coeff_vec, vertex)) + bias for vertex in vertices]
        return float(min(values)), float(max(values))

    def _enumerate_vertices(self, atol: float) -> List[Vector]:
        vertices: List[Vector] = []
        rows = list(range(len(self.A)))
        for idxs in combinations(rows, self.dimension):
            Ai = tuple(self.A[i] for i in idxs)
            if matrix_rank(Ai, atol) < self.dimension:
                continue
            bi = tuple(self.b[i] for i in idxs)
            try:
                sol = solve(Ai, bi, atol)
            except ValueError:
                continue
            if self.contains(sol, atol):
                vertices.append(sol)
        return vertices

    def project_box(self) -> Tuple[Vector, Vector]:
        lowers: List[float] = []
        uppers: List[float] = []
        for dim in range(self.dimension):
            coeff = [0.0] * self.dimension
            coeff[dim] = 1.0
            lb, ub = self.bounds_on_linear_form(coeff, 0.0)
            lowers.append(lb)
            uppers.append(ub)
        return tuple(lowers), tuple(uppers)

    @staticmethod
    def from_bounds(lower: Sequence[float], upper: Sequence[float]) -> "Polytope":
        lower_vec = as_vector(lower)
        upper_vec = as_vector(upper)
        if len(lower_vec) != len(upper_vec):
            raise ValueError("shape mismatch")
        rows: List[Vector] = []
        bounds: List[float] = []
        for i in range(len(lower_vec)):
            unit = [0.0] * len(lower_vec)
            unit[i] = 1.0
            rows.append(tuple(unit))
            bounds.append(upper_vec[i])
            unit[i] = -1.0
            rows.append(tuple(unit))
            bounds.append(-lower_vec[i])
        return Polytope(tuple(rows), tuple(bounds))

