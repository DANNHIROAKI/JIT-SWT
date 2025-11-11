from __future__ import annotations

import math

from jitswt.jit import BranchAndBoundAnalyzer, JITBranchAndBound
from jitswt.network import NetworkBuilder
from jitswt.polytope import Polytope


def build_simple_network() -> NetworkBuilder:
    builder = NetworkBuilder(input_dim=2)
    builder.add_affine([[1.0, -1.0], [0.5, 0.5]], [0.0, 0.0])
    builder.add_relu(2)
    builder.add_affine([[2.0, -1.0]], [0.0])
    return builder


def test_evaluate_matches_expected_value():
    builder = build_simple_network()
    domain = Polytope.from_bounds([-1.0, -1.0], [1.0, 1.0])
    net = builder.build(domain)
    y = net.evaluate([0.5, -0.2])
    assert y == (1.25,)


def test_enumerate_pieces_covers_domain():
    builder = build_simple_network()
    domain = Polytope.from_bounds([-1.0, -1.0], [1.0, 1.0])
    net = builder.build(domain)
    pieces = net.enumerate_pieces()
    x = (0.2, -0.4)
    y = net.evaluate(x)
    for piece in pieces:
        if piece.polytope.contains(x):
            assert piece.evaluate(x) == y
            break
    else:  # pragma: no cover - failure for debugging
        raise AssertionError("point not covered by any piece")


def test_branch_and_bound_maximization_bounds_are_consistent():
    builder = build_simple_network()
    domain = Polytope.from_bounds([-1.0, -1.0], [1.0, 1.0])
    net = builder.build(domain)
    pieces = net.enumerate_pieces()
    analyzer = BranchAndBoundAnalyzer(pieces)
    result = analyzer.maximize([1.0])
    assert result.certificate in pieces
    assert result.status == "complete"
    assert result.lower_bound <= result.upper_bound


def test_piecewise_lipschitz_is_finite():
    builder = build_simple_network()
    domain = Polytope.from_bounds([-1.0, -1.0], [1.0, 1.0])
    net = builder.build(domain)
    pieces = net.enumerate_pieces()
    analyzer = BranchAndBoundAnalyzer(pieces)
    lipschitz = analyzer.piecewise_lipschitz(2.0)
    assert math.isfinite(lipschitz)
    assert lipschitz >= 0


def test_jit_branch_and_bound_matches_static_bounds():
    builder = build_simple_network()
    domain = Polytope.from_bounds([-1.0, -1.0], [1.0, 1.0])
    net = builder.build(domain)
    static = BranchAndBoundAnalyzer(net.enumerate_pieces()).maximize([1.0])
    dynamic = JITBranchAndBound(net).maximize([1.0])
    assert dynamic.status == "complete"
    assert dynamic.upper_bound == static.upper_bound
    assert dynamic.lower_bound == static.lower_bound


def test_jit_branch_and_bound_budget():
    builder = build_simple_network()
    domain = Polytope.from_bounds([-1.0, -1.0], [1.0, 1.0])
    net = builder.build(domain)
    result = JITBranchAndBound(net).maximize([1.0], budget=0)
    assert result.status == "budget"
    assert result.explored_pieces == 0


def test_abs_layer_refinement():
    builder = NetworkBuilder(input_dim=1)
    builder.add_affine([[1.0]], [0.0])
    builder.add_abs(1)
    domain = Polytope.from_bounds([-2.0], [2.0])
    net = builder.build(domain)
    pieces = net.enumerate_pieces()
    outputs = {piece.evaluate([-1.5])[0] for piece in pieces if piece.polytope.contains([-1.5])}
    assert outputs == {1.5}


def test_prelu_layer_negative_slope():
    builder = NetworkBuilder(input_dim=1)
    builder.add_affine([[1.0]], [0.0])
    builder.add_prelu([0.25])
    domain = Polytope.from_bounds([-2.0], [2.0])
    net = builder.build(domain)
    pieces = net.enumerate_pieces()
    neg_outputs = {piece.evaluate([-2.0])[0] for piece in pieces if piece.polytope.contains([-2.0])}
    assert neg_outputs == {-0.5}
    pos_outputs = {piece.evaluate([1.0])[0] for piece in pieces if piece.polytope.contains([1.0])}
    assert pos_outputs == {1.0}

