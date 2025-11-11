from jitswt import BranchAndBoundAnalyzer, JITBranchAndBound, NetworkBuilder
from jitswt.polytope import Polytope


def build_simple_network():
    builder = NetworkBuilder(input_dim=2)
    builder.add_affine([[1.0, -1.0], [0.5, 0.5]], [0.0, 0.0])
    builder.add_relu(2)
    builder.add_affine([[2.0, -1.0]], [0.0])
    domain = Polytope.from_bounds([-1.0, -1.0], [1.0, 1.0])
    return builder.build(domain)


def test_evaluate_matches_expected_value():
    net = build_simple_network()
    y = net.evaluate([0.5, -0.2])
    assert y == (1.25,)


def test_enumerate_pieces_covers_domain():
    net = build_simple_network()
    pieces = net.enumerate_pieces()
    x = (0.2, -0.4)
    y = net.evaluate(x)
    for piece in pieces:
        if piece.polytope.contains(x):
            assert piece.evaluate(x) == y
            break
    else:  # pragma: no cover - failure for debugging
        raise AssertionError("point not covered by any piece")


def test_branch_and_bound_maximization():
    net = build_simple_network()
    pieces = net.enumerate_pieces()
    analyzer = BranchAndBoundAnalyzer(pieces)
    result = analyzer.maximize([1.0])
    assert result.certificate in pieces
    assert result.status == "complete"
    assert result.upper_bound == result.lower_bound


def test_piecewise_lipschitz_is_non_negative():
    net = build_simple_network()
    pieces = net.enumerate_pieces()
    analyzer = BranchAndBoundAnalyzer(pieces)
    lipschitz = analyzer.piecewise_lipschitz(2.0)
    assert lipschitz >= 0


def test_jit_branch_and_bound_matches_static():
    net = build_simple_network()
    static = BranchAndBoundAnalyzer(net.enumerate_pieces()).maximize([1.0])
    dynamic = JITBranchAndBound(net).maximize([1.0])
    assert dynamic.status == "complete"
    assert dynamic.upper_bound == static.upper_bound
    assert dynamic.lower_bound == static.lower_bound


def test_jit_branch_and_bound_budget():
    net = build_simple_network()
    result = JITBranchAndBound(net).maximize([1.0], budget=0)
    assert result.status == "budget"
    # Budget 0 should not discover any piece
    assert result.explored_pieces == 0

