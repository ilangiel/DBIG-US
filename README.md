# DBIG-US: A two-stage under-sampling algorithm to face the class imbalance problem

> **Article:** Guzmán-Ponce, A., et al. (2021). *DBIG-US: A two-stage under-sampling algorithm to face the class imbalance problem in big data*. **Expert Systems With Applications**, 168, 114301. [DOI: 10.1016/j.eswa.2020.114301](https://doi.org/10.1016/j.eswa.2020.114301)

## 📖 Algorithm Overview

**DBIG-US** is a two-stage under-sampling method designed for class-imbalanced binary datasets. It does **not** require specifying the number of clusters in advance and lets the user set a maximum imbalance ratio (IR) for the output.

```
Input:  DS = {p1, p2, …, pn},  maxIR
Output: DS_balanced

1. Split DS into C⁻ (majority) and C⁺ (minority)
2. C1⁻ ← C⁻
3. repeat
4.     Compute ε and minPts
5.     C1⁻ ← DBSCAN(C1⁻, ε, minPts)
6. until ε and minPts do not change
7. C2⁻ ← ShapeGraph(C1⁻, Rmax, C⁺)
8. DS_balanced ← C⁺ ∪ C2⁻
```

### Stage 1 — Filtering (DBSCAN)

DBSCAN is applied **only on the majority class** with automatically estimated parameters:

| Parameter | Formula |
|-----------|---------|
| ε | Average Euclidean distance of all majority instances to their centroid |
| minPts | `(π·ε²) / ((4/3)·π·ε³) × |C⁺|` |

Instances identified as **noise are removed** — they are borderline or irrelevant for learning.

### Stage 2 — Under-Sampling (ShapeGraph)

A complete weighted graph `Gw` is built on the noise-free majority class `C1⁻`:
- Vertices = majority instances
- Edge weight = Euclidean distance between instances

The algorithm iteratively picks pairs of instances with **maximum distance** (most dispersed), adding unvisited endpoints to `C2⁻` until `|C2⁻| / |C⁺| ≤ maxIR`.

The target size is reduced using:

```
Sample = (|C1⁻| · σ² · Z²) / (e²·(|C1⁻|−1) + σ²·Z²)
```

with Z=1.96, σ=0.5, e=0.05 (95 % confidence).

## 🚀 Quick Start

```bash
# Install dependencies
pip install numpy matplotlib

# Run the demo
python dbig_us.py
```

## 📊 What the Demo Does

1. Generates a synthetic imbalanced dataset (200 majority / 20 minority instances + noise).
2. Applies DBIG-US with `maxIR = 1.5`.
3. Prints majority/minority counts before and after each stage.
4. Saves a side-by-side scatter plot as **`dbig_us_result.png`**.

## 🗂 File Structure

```
demo_DBIG_US/
├── dbig_us.py          # Complete implementation + demo
└── README.md           # This file
```

## 📐 Key Functions

| Function | Description |
|----------|-------------|
| `compute_epsilon_minpts(C_neg, C_pos)` | Estimates DBSCAN parameters (Eqs. 1 & 2) |
| `dbscan_filter(C_neg, ε, minPts)` | Removes noisy majority instances |
| `shape_graph(C1_neg, C_pos, maxIR)` | Graph-based under-sampling |
| `dbig_us(DS, maxIR)` | Main algorithm (Algorithm 1) |

## 📋 Dependencies

| Library | Purpose |
|---------|---------|
| `numpy` | Numerical computation |
| `matplotlib` | Visualisation |
