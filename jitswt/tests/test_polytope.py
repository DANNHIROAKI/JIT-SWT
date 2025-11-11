import numpy as np

from jitswt.polytope import Polytope


def test_bounds_on_linear_form_box():
    poly = Polytope.from_bounds([-1, -2], [1, 2])
    coeff = np.array([1.0, -1.0])
    lb, ub = poly.bounds_on_linear_form(coeff, 0.5)
    assert np.isclose(lb, -2.5)
    assert np.isclose(ub, 3.5)


def test_contains_point():
    poly = Polytope.from_bounds([0, 0], [1, 1])
    assert poly.contains([0.5, 0.5])
    assert not poly.contains([1.5, 0.5])


def test_feasibility_checks():
    base = Polytope.from_bounds([-1, -1], [1, 1])
    assert base.is_feasible()
    infeasible = base.intersection_with_halfspace([1.0, 0.0], -2.0)
    assert not infeasible.is_feasible()
