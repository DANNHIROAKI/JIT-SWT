"""JIT-SWT: Just-in-time symbolic weighted transducers for CPWL networks.

This package implements the algorithms described in the accompanying paper.
"""

from .guard import GuardLibrary, GuardSet
from .polytope import Polytope
from .layers import AffineLayer, ReLULayer, LeakyReLULayer, MaxLayer
from .network import CPWLNetwork, NetworkBuilder
from .jit import BranchAndBoundAnalyzer, JITBranchAndBound, MaximizationResult

__all__ = [
    "GuardLibrary",
    "GuardSet",
    "Polytope",
    "AffineLayer",
    "ReLULayer",
    "LeakyReLULayer",
    "MaxLayer",
    "CPWLNetwork",
    "NetworkBuilder",
    "BranchAndBoundAnalyzer",
    "JITBranchAndBound",
    "MaximizationResult",
]
