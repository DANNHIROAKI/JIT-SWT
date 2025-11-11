from jitswt.polytope import Polytope


def test_polytope_contains_and_bounds():
    poly = Polytope.from_bounds([-1.0, -1.0], [1.0, 2.0])
    assert poly.contains([0.0, 0.0])
    assert not poly.contains([1.5, 0.0])
    lb, ub = poly.bounds_on_linear_form([1.0, -1.0])
    assert lb <= -3.0 <= ub
    assert lb <= 2.0 <= ub

