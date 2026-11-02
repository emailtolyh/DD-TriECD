# DD-TriECD# DD-TriECD

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Research-yellow)

**Dynamic Directed Triple Evolutionary Community Detection** with Federated Representation Learning.

DD-TriECD is a research-oriented framework for detecting evolving community structures in multi-layer attributed networks under organizational hierarchy and role constraints. It combines federated graph representation learning, evolutionary multi-objective optimization, and constraint-aware coarsening to produce deployable partitions across network snapshots.

## Table of Contents

- [Key Features](#key-features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Input Format](#input-format)
- [Output](#output)
- [Configuration Parameters](#configuration-parameters)
- [Methodology Overview](#methodology-overview)
- [Privacy Note](#privacy-note)
- [Limitations](#limitations)
- [Citation](#citation)
- [License](#license)

---

## Key Features

- **Multi-layer dynamic community detection** over attributed undirected networks with evolving nodes and edge weights
- **Three-objective optimization**: intra-layer modularity, cross-layer agreement, and temporal smoothness
- **Hierarchy and role constraints**: enforces organizational structure similarity thresholds and business unit role templates
- **Federated embedding learning**: client-local GCN training with server parameter averaging, gradient clipping, and Gaussian noise
- **Super-node coarsening**: constraint-preserving graph reduction with warm-start reuse from previous snapshots
- **NSGA-III environmental selection**: Pareto-front approximation with reference direction survival
- **Adaptive preference weights**: multiplicative weight update based on achievement scalarization deficits
- **Consensus partitioning**: Hungarian-aligned cross-layer correspondence with majority voting and constraint auditing

## Installation

```bash
# (Recommended) Create a virtual environment
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate

# Install dependencies
pip install numpy torch scipy scikit-learn pymoo


from ddtriecd import DDTriECD, Config, make_demo_snapshot

config = Config(
    seed=42,
    gamma=0.25,
    tau_H=0.8,
    delta_R=0.10,
    population_size=24,
    generations=8,
    rounds=2,
)

snapshots = [
    make_demo_snapshot(scale=1.0),
    make_demo_snapshot(scale=1.2),
]

algorithm = DDTriECD(config)
results = algorithm.fit(snapshots)

for t, result in enumerate(results, 1):
    print(f"Snapshot {t}: {result['status']}")
    print(f"Consensus labels: {result['consensus_labels']}")



Methodology Overview
1. Federated Embedding Learning
Each client trains a local GCN on its induced subgraph. Per-anchor gradient clipping and Gaussian noise are applied before server aggregation. Embeddings capture structural and attribute information without sharing raw data across clients.
2. Constraint-Aware Coarsening
Nodes are greedily merged into super-nodes when hierarchy similarity and role template constraints are satisfied. Previous snapshot groupings serve as warm starts; significant changes trigger re-coarsening.
3. Evolutionary Multi-Objective Search
NSGA-III optimizes three objectives simultaneously. Crossover uses Hungarian matching for label alignment. Mutation and repair operators enforce hierarchy and role constraints while minimizing violations.
4. Cross-Layer Consensus
Communities are aligned across layers via Hungarian matching. A consensus partition is obtained through weighted majority voting and audited against all constraints.
Privacy Note

The gradient clipping and noise mechanism simulates federated training but does not provide certified node-level differential privacy. It is intended for centralized research simulation only.

Limitations

Undirected graphs only (GCN uses symmetric normalization)
Consistent layer count across snapshots
Equal feature dimensions across layers
No theoretical guarantee of finding feasible solutions
