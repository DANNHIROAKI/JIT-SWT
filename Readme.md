# JIT-SWT: Just-In-Time Piecewise-Linear Semantics for ReLU-type Networks

This repository contains a lightweight reference implementation of the
techniques described in the research manuscript.  The code provides tools for
constructing piecewise-linear neural networks, compiling them into shared guard
structures, and performing basic geometric and verification queries.

## Features

* Guard library that stores unique half-space constraints in normalised form.
* Polytope utilities capable of computing bounds using deterministic vertex
  enumeration (no third-party solver required).
* Layer abstractions for affine transforms, ReLU/Leaky-ReLU/PReLU activations,
  absolute-value gates, and pairwise max pooling.
* Network builder that compiles a sequential DAG into a shared-guard JIT graph
  and can enumerate linear pieces on demand.
* Branch-and-bound analyzers that report both upper and lower certificates and
  operate either on the full piece set or by refining regions lazily with a
  configurable budget.

## Quickstart

```python
from jitswt import (
    NetworkBuilder,
    BranchAndBoundAnalyzer,
    JITBranchAndBound,
)
from jitswt.polytope import Polytope

builder = NetworkBuilder(input_dim=2)
builder.add_affine([[1.0, -1.0], [0.5, 0.5]], [0.0, 0.0])
builder.add_relu(2)
builder.add_affine([[2.0, -1.0]], [0.0])
network = builder.build(Polytope.from_bounds([-1, -1], [1, 1]))

# Enumerate pieces explicitly (useful for offline inspection)
pieces = network.enumerate_pieces()
static = BranchAndBoundAnalyzer(pieces)
print("Number of linear pieces:", len(pieces))
print("Lipschitz constant (2-norm):", static.piecewise_lipschitz())

# Alternatively, drive analysis directly from the network with JIT refinement
dynamic = JITBranchAndBound(network)
result = dynamic.maximize([1.0])
print("Maximum value:", result.upper_bound)
```

Run the unit tests with `pytest`:

```bash
pip install -r requirements.txt
pytest -q
```
