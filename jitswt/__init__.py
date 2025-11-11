"""JIT-SWT: Just-in-time symbolic weighted transducers for CPWL networks.

This package implements the algorithms described in the accompanying paper.
"""

from .guard import GuardLibrary, GuardSet
from .polytope import Polytope
from .layers import AbsLayer, AffineLayer, LeakyReLULayer, MaxLayer, PReLULayer, ReLULayer
from .network import CPWLNetwork, NetworkBuilder
from .jit import BranchAndBoundAnalyzer, JITBranchAndBound, MaximizationResult

__all__ = [
    "GuardLibrary",
    "GuardSet",
    "Polytope",
    "AbsLayer",
    "AffineLayer",
    "ReLULayer",
    "LeakyReLULayer",
    "PReLULayer",
    "MaxLayer",
    "CPWLNetwork",
    "NetworkBuilder",
    "BranchAndBoundAnalyzer",
    "JITBranchAndBound",
    "MaximizationResult",
]
