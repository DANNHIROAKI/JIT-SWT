import numpy as np

from jitswt import BranchAndBoundAnalyzer, NetworkBuilder
from jitswt.polytope import Polytope


def build_simple_network():
    builder = NetworkBuilder(input_dim=2)
    builder.add_affine(np.array([[1.0, -1.0], [0.5, 0.5]]), np.array([0.0, 0.0]))
    builder.add_relu(2)
    builder.add_affine(np.array([[2.0, -1.0]]), np.array([0.0]))
    domain = Polytope.from_bounds([-1, -1], [1, 1])
    return builder.build(domain)


def test_evaluate_matches_numpy():
    net = build_simple_network()
    x = np.array([0.5, -0.2])
    y = net.evaluate(x)
    expected = np.array([0.0])
    assert np.allclose(y, expected)


def test_enumerate_pieces():
    net = build_simple_network()
    pieces = net.enumerate_pieces()
    assert len(pieces) >= 1
    x = np.array([0.2, -0.4])
    y = net.evaluate(x)
    for piece in pieces:
        if piece.polytope.contains(x):
            assert np.allclose(piece.evaluate(x), y)
            break
    else:
        raise AssertionError("point not covered by any piece")


def test_branch_and_bound_maximization():
    net = build_simple_network()
    pieces = net.enumerate_pieces()
    analyzer = BranchAndBoundAnalyzer(pieces)
    coeff = np.array([1.0])
    max_value, piece = analyzer.maximize(coeff)
    xs = [piece.polytope.bounds_on_linear_form(np.array([1.0, 0.0]))[1],
          piece.polytope.bounds_on_linear_form(np.array([0.0, 1.0]))[1]]
    assert max_value >= piece.evaluate(np.array(xs)).item() - 1e-6


def test_piecewise_lipschitz():
    net = build_simple_network()
    pieces = net.enumerate_pieces()
    analyzer = BranchAndBoundAnalyzer(pieces)
    lipschitz = analyzer.piecewise_lipschitz(2.0)
    assert lipschitz >= 0
