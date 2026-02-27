"""
DBIG-US: A Two-Stage Under-Sampling Algorithm to Face the Class Imbalance Problem in Big Data
==============================================================================================
Reference: Guzmán-Ponce et al. (2021). Expert Systems With Applications, 168, 114301.

This demo implements the DBIG-US algorithm, which uses:
  1. Filtering Step  — DBSCAN to remove noisy majority-class instances.
  2. Under-sampling Step — ShapeGraph to reduce the majority class to reach
     a target imbalance ratio (IR ≤ maxIR).

Paper Algorithm 1 — DBIG-US
Paper Algorithm 2 — ShapeGraph
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


# ─────────────────────────────────────────────────────────────────────────────
# Euclidean distance
# ─────────────────────────────────────────────────────────────────────────────
def euclidean(a, b):
    return np.sqrt(np.sum((a - b) ** 2))


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — DBSCAN (adapted from article equations 1 & 2)
# ─────────────────────────────────────────────────────────────────────────────
def compute_epsilon_minpts(C_neg, C_pos):
    """
    Compute epsilon and minPts as defined in the paper (Eqs. 1 & 2).
    """
    m = C_neg.mean(axis=0)
    dists = np.array([euclidean(m, p) for p in C_neg])
    # Paper Eq. 1: sqrt of the sum divided by |C-|
    epsilon = np.sqrt(dists.sum()) / len(C_neg)
    # Ensure epsilon is at least the median nearest-neighbor distance
    # (prevents degenerate near-zero ε that erases all points)
    nn_dists = sorted(dists)
    epsilon = max(epsilon, nn_dists[len(nn_dists) // 4])  # at least 25th percentile

    total_volume = (4 / 3) * np.pi * (epsilon ** 3)
    area = np.pi * (epsilon ** 2)
    minPts = max(1, int((area / total_volume) * len(C_pos)))
    return epsilon, minPts


def dbscan_filter(C_neg, epsilon, minPts):
    """
    Apply DBSCAN on the majority class only and return noise-free instances.
    Instances labelled as noise are removed (Algorithm 1 in the paper).
    """
    n = len(C_neg)
    visited = [False] * n
    labels = [-1] * n          # -1 = unassigned  0+ = cluster id   -2 = noise
    cluster_id = 0

    def get_neighbors(idx):
        return [j for j in range(n) if j != idx and
                euclidean(C_neg[idx], C_neg[j]) <= epsilon]

    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True
        neighbors = get_neighbors(i)
        if len(neighbors) < minPts:
            labels[i] = -2          # noise
        else:
            labels[i] = cluster_id
            seeds = set(neighbors)
            while seeds:
                j = seeds.pop()
                if not visited[j]:
                    visited[j] = True
                    nb = get_neighbors(j)
                    if len(nb) >= minPts:
                        seeds.update(nb)
                if labels[j] == -1:
                    labels[j] = cluster_id
            cluster_id += 1

    # Keep only non-noise instances
    clean = C_neg[[i for i, lbl in enumerate(labels) if lbl >= 0]]
    return clean


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — ShapeGraph (Algorithm 2 in the paper)
# ─────────────────────────────────────────────────────────────────────────────
def compute_sample_size(n, iteration=1, Z=1.96, sigma=0.5, e=0.05):
    """
    Iterative sample-size formula (Eq. 3 in the paper).
    Reduces the target size at each call.
    """
    numerator = n * (sigma ** 2) * (Z ** 2)
    denominator = (e ** 2) * (n - 1) + (sigma ** 2) * (Z ** 2)
    size = int(numerator / denominator)
    for _ in range(iteration - 1):
        n = size
        numerator = n * (sigma ** 2) * (Z ** 2)
        denominator = (e ** 2) * (n - 1) + (sigma ** 2) * (Z ** 2)
        size = int(numerator / denominator)
    return max(1, size)


def shape_graph(C1_neg, C_pos, max_IR):
    """
    Build a complete weighted graph on C1- and iteratively select the pair of
    instances with maximum distance, adding unvisited endpoints to C2-, until
    IR = |C2-| / |C+| <= maxIR.
    """
    n = len(C1_neg)
    n_pos = len(C_pos)
    target = int(n_pos * max_IR)
    target = min(target, n)

    # Build distance matrix
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = euclidean(C1_neg[i], C1_neg[j])
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d

    visited = [False] * n
    C2_neg_idx = []

    while len(C2_neg_idx) < target:
        # Find the edge (pair) with maximum weight among unvisited nodes
        best_i, best_j, best_d = -1, -1, -1
        for i in range(n):
            for j in range(i + 1, n):
                if dist_matrix[i, j] > best_d:
                    best_d = dist_matrix[i, j]
                    best_i, best_j = i, j

        for idx in [best_i, best_j]:
            if not visited[idx] and len(C2_neg_idx) < target:
                visited[idx] = True
                C2_neg_idx.append(idx)

        # Zero out this edge so it isn't picked again
        dist_matrix[best_i, best_j] = 0
        dist_matrix[best_j, best_i] = 0

        if best_d == 0:          # no more edges
            break

    return C1_neg[C2_neg_idx]


# ─────────────────────────────────────────────────────────────────────────────
# Main — DBIG-US  (Algorithm 1 in the paper)
# ─────────────────────────────────────────────────────────────────────────────
def dbig_us(DS, max_IR=1.5):
    """
    DBIG-US main function.

    Parameters
    ----------
    DS      : numpy ndarray  shape (n, d+1)  last column is the class label
    max_IR  : float  maximum imbalance ratio allowed in the output

    Returns
    -------
    DS_balanced : numpy ndarray  balanced data set
    """
    labels = DS[:, -1]
    features = DS[:, :-1]

    # Identify minority (+) and majority (-) classes
    classes, counts = np.unique(labels, return_counts=True)
    minority_label = classes[np.argmin(counts)]
    majority_label = classes[np.argmax(counts)]

    C_pos = features[labels == minority_label]
    C_neg = features[labels == majority_label]

    print(f"[DBIG-US] Original: {len(C_neg)} majority / {len(C_pos)} minority  "
          f"(IR = {len(C_neg)/len(C_pos):.2f})")

    # ── Stage 1: Filtering (DBSCAN loop until params stabilise) ──────────────
    prev_epsilon, C1_neg = None, C_neg.copy()
    for _ in range(20):          # max 20 iterations
        epsilon, minPts = compute_epsilon_minpts(C1_neg, C_pos)
        if prev_epsilon is not None and abs(epsilon - prev_epsilon) < 1e-6:
            break
        candidate = dbscan_filter(C1_neg, epsilon, minPts)
        # Guard: never let DBSCAN erase the entire majority class
        if len(candidate) == 0:
            break
        C1_neg = candidate
        prev_epsilon = epsilon

    print(f"[DBIG-US] After DBSCAN filter: {len(C1_neg)} majority instances")

    # ── Stage 2: ShapeGraph under-sampling ───────────────────────────────────
    if len(C1_neg) > 0:
        C2_neg = shape_graph(C1_neg, C_pos, max_IR)
    else:
        C2_neg = C1_neg

    print(f"[DBIG-US] After ShapeGraph: {len(C2_neg)} majority instances  "
          f"(IR = {len(C2_neg)/len(C_pos):.2f})")

    # Reconstruct balanced data set
    pos_rows = np.column_stack([C_pos, np.full(len(C_pos), minority_label)])
    neg_rows = np.column_stack([C2_neg, np.full(len(C2_neg), majority_label)])
    DS_balanced = np.vstack([pos_rows, neg_rows])
    return DS_balanced


# ─────────────────────────────────────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────────────────────────────────────
def make_imbalanced_dataset(n_majority=200, n_minority=20, seed=42):
    rng = np.random.default_rng(seed)
    majority = rng.multivariate_normal([2, 2], [[1, 0], [0, 1]], n_majority)
    minority = rng.multivariate_normal([0, 0], [[0.3, 0], [0, 0.3]], n_minority)
    # add some noise to majority
    noise = rng.multivariate_normal([5, 5], [[0.5, 0], [0, 0.5]], 10)
    majority = np.vstack([majority, noise])

    X = np.vstack([majority, minority])
    y = np.concatenate([np.zeros(len(majority)), np.ones(len(minority))])
    return np.column_stack([X, y])


if __name__ == "__main__":
    DS = make_imbalanced_dataset()
    DS_balanced = dbig_us(DS, max_IR=1.5)

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("DBIG-US: Two-Stage Under-Sampling",
                 fontsize=12)

    for ax, data, title in zip(
            axes,
            [DS, DS_balanced],
            ["Original imbalanced dataset", "After DBIG-US (balanced)"]):
        maj = data[data[:, -1] == 0, :2]
        minn = data[data[:, -1] == 1, :2]
        ax.scatter(maj[:, 0], maj[:, 1], c="#e74c3c", alpha=0.5,
                   label=f"Majority ({len(maj)})", edgecolors="none")
        ax.scatter(minn[:, 0], minn[:, 1], c="#2ecc71", alpha=0.8,
                   label=f"Minority ({len(minn)})", marker="^", s=80)
        ax.set_title(title)
        ax.legend()
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")

    plt.tight_layout()
    plt.savefig("dbig_us_result.png", dpi=150)
    plt.show()
    print("\nPlot saved as dbig_us_result.png")
