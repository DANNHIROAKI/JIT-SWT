# JIT-SWT: Just-In-Time Piecewise-Linear Semantics for ReLU-type Networks

This repository contains a lightweight reference implementation of the
techniques described in the research manuscript.  The code provides tools for
constructing piecewise-linear neural networks, compiling them into shared guard
structures, and performing basic geometric and verification queries.

## Features

* Guard library that stores unique half-space constraints in normalised form.
* Polytope utilities capable of computing bounds using deterministic vertex
  enumeration (no third-party solver required).
* Layer abstractions for affine transforms, ReLU/Leaky-ReLU activations, and
  pairwise max gates.
* Network builder that compiles a sequential DAG into an explicit collection of
  linear pieces with shared guards.
* Branch-and-bound style analyzer offering maximisation and Lipschitz
  computations over the enumerated pieces.

## Quickstart

```python
from jitswt import NetworkBuilder, BranchAndBoundAnalyzer
from jitswt.polytope import Polytope

builder = NetworkBuilder(input_dim=2)
builder.add_affine([[1.0, -1.0], [0.5, 0.5]], [0.0, 0.0])
builder.add_relu(2)
builder.add_affine([[2.0, -1.0]], [0.0])
network = builder.build(Polytope.from_bounds([-1, -1], [1, 1]))

pieces = network.enumerate_pieces()
analyzer = BranchAndBoundAnalyzer(pieces)
print("Number of linear pieces:", len(pieces))
print("Lipschitz constant (2-norm):", analyzer.piecewise_lipschitz())
```

Run the unit tests with `pytest`:

```bash
pip install -r requirements.txt
pytest -q
```
