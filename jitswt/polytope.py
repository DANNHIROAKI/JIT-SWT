"""Basic polytope utilities used by the JIT-SWT implementation."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, Sequence, Tuple

import numpy as np

try:  # Optional SciPy dependency used when available
    from scipy.optimize import linprog  # type: ignore
except Exception:  # pragma: no cover - SciPy is optional
    linprog = None  # type: ignore


@dataclass
class Polytope:
    """Convex polytope :math:`\{x \mid Ax \le b\}`."""

    A: np.ndarray
    b: np.ndarray

    def __post_init__(self) -> None:
        if self.A.ndim != 2:
            raise ValueError("A must be a 2D array")
        if self.b.ndim != 1:
            raise ValueError("b must be a 1D array")
        if self.A.shape[0] != self.b.shape[0]:
            raise ValueError("A and b shapes incompatible")
        self._dim = self.A.shape[1]

    @property
    def dimension(self) -> int:
        return self._dim

    def intersection_with_halfspace(self, normal: Iterable[float], offset: float) -> "Polytope":
        normal_arr = np.asarray(list(normal), dtype=float)
        if normal_arr.shape != (self.dimension,):
            raise ValueError("dimension mismatch")
        new_A = np.vstack([self.A, normal_arr])
        new_b = np.concatenate([self.b, [float(offset)]])
        return Polytope(new_A, new_b)

    def is_feasible(self, atol: float = 1e-9) -> bool:
        """Return ``True`` iff the polytope contains at least one point."""

        if linprog is not None:
            objective = np.zeros(self.dimension)
            bounds = [(None, None)] * self.dimension
            res = linprog(
                objective,
                A_ub=self.A,
                b_ub=self.b,
                bounds=bounds,
                method="highs",
            )
            if res.success:
                return True
        vertices = self._enumerate_vertices(atol)
        return len(vertices) > 0

    def contains(self, point: Sequence[float], atol: float = 1e-9) -> bool:
        x = np.asarray(list(point), dtype=float)
        if x.shape != (self.dimension,):
            raise ValueError("dimension mismatch")
        return bool(np.all(self.A @ x <= self.b + atol))

    def bounds_on_linear_form(self, coeff: Iterable[float], bias: float = 0.0, atol: float = 1e-9) -> Tuple[float, float]:
        coeff_arr = np.asarray(list(coeff), dtype=float)
        if coeff_arr.shape != (self.dimension,):
            raise ValueError("dimension mismatch")
        bias = float(bias)
        if linprog is not None:
            bounds = [(None, None)] * self.dimension
            res_max = linprog(-coeff_arr, A_ub=self.A, b_ub=self.b, bounds=bounds)  # type: ignore[arg-type]
            res_min = linprog(coeff_arr, A_ub=self.A, b_ub=self.b, bounds=bounds)  # type: ignore[arg-type]
            if res_max.success and res_min.success:
                # ``linprog`` returns the value of the minimised objective.  For the
                # maximum we minimise ``-coeff @ x`` and negate the optimum.  For the
                # minimum we minimise ``coeff @ x`` directly.  In both cases the
                # constant ``bias`` must be added to obtain the affine form bounds.
                return (res_min.fun + bias, -res_max.fun + bias)
        # Fallback vertex enumeration
        vertices = self._enumerate_vertices(atol)
        if not vertices:
            raise RuntimeError("polytope appears empty or unbounded")
        values = [coeff_arr @ v + bias for v in vertices]
        return float(min(values)), float(max(values))

    def _enumerate_vertices(self, atol: float) -> Sequence[np.ndarray]:
        m, n = self.A.shape
        vertices = []
        for idxs in combinations(range(m), n):
            Ai = self.A[list(idxs), :]
            if np.linalg.matrix_rank(Ai) < n:
                continue
            bi = self.b[list(idxs)]
            try:
                sol = np.linalg.solve(Ai, bi)
            except np.linalg.LinAlgError:
                continue
            if np.all(self.A @ sol <= self.b + atol):
                vertices.append(sol)
        return vertices

    def project_box(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return per-dimension min/max bounds by solving :math:`2n` LPs."""

        lowers = []
        uppers = []
        for i in range(self.dimension):
            coeff = np.zeros(self.dimension)
            coeff[i] = 1.0
            lb, ub = self.bounds_on_linear_form(coeff, 0.0)
            lowers.append(lb)
            uppers.append(ub)
        return np.array(lowers), np.array(uppers)

    @staticmethod
    def from_bounds(lower: Sequence[float], upper: Sequence[float]) -> "Polytope":
        lower_arr = np.asarray(list(lower), dtype=float)
        upper_arr = np.asarray(list(upper), dtype=float)
        if lower_arr.shape != upper_arr.shape:
            raise ValueError("shapes mismatch")
        A = []
        b = []
        dim = lower_arr.shape[0]
        for i in range(dim):
            row = np.zeros(dim)
            row[i] = 1.0
            A.append(row)
            b.append(upper_arr[i])
            row = np.zeros(dim)
            row[i] = -1.0
            A.append(row)
            b.append(-lower_arr[i])
        return Polytope(np.asarray(A), np.asarray(b))
