from jitswt import BranchAndBoundAnalyzer, NetworkBuilder
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
    value, piece = analyzer.maximize([1.0])
    assert isinstance(value, float)
    assert piece in pieces


def test_piecewise_lipschitz_is_non_negative():
    net = build_simple_network()
    pieces = net.enumerate_pieces()
    analyzer = BranchAndBoundAnalyzer(pieces)
    lipschitz = analyzer.piecewise_lipschitz(2.0)
    assert lipschitz >= 0

