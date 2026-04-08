"""
correlation_graph.py
====================
Financial Correlation Network Builder

Builds undirected weighted graphs from asset return correlations using:
  - Pearson correlation (linear dependencies)
  - Spearman rank correlation (monotonic / non-linear)
  - Partial correlation (direct dependencies, removing confounders)

Advanced features:
  - Threshold-based edge filtering
  - Graph sparsification (PMFG, MST, random-walk sparsification)
  - Spectral graph learning (GL-SigRep / precision-matrix approach)
  - Statistical significance filtering (p-value based)

Journal reference:
  Mantegna (1999) "Hierarchical structure in financial markets"
  Eur. Phys. J. B 11, 193-197
"""

from __future__ import annotations

import warnings
from typing import Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats
from scipy.linalg import inv
from sklearn.covariance import GraphicalLassoCV, LedoitWolf


# ─────────────────────────────────────────────────────────────
# 1. Core Correlation Graph Builders
# ─────────────────────────────────────────────────────────────

def build_pearson_graph(
    returns: pd.DataFrame,
    threshold: float = 0.3,
    p_value_cutoff: float = 0.05,
    min_periods: int = 30,
) -> nx.Graph:
    """
    Build a correlation network using Pearson correlation.

    Parameters
    ----------
    returns : pd.DataFrame
        Asset returns matrix (rows = time, columns = assets).
    threshold : float
        Minimum |correlation| to include an edge.
    p_value_cutoff : float
        Maximum p-value for statistical significance filter.
    min_periods : int
        Minimum overlapping observations required per pair.

    Returns
    -------
    nx.Graph
        Weighted undirected graph. Edge attribute: 'weight' (correlation).
    """
    assets = returns.columns.tolist()
    n = len(assets)
    G = nx.Graph()
    G.add_nodes_from(assets)

    for i in range(n):
        for j in range(i + 1, n):
            x = returns.iloc[:, i].dropna()
            y = returns.iloc[:, j].dropna()
            # align
            xy = pd.concat([x, y], axis=1).dropna()
            if len(xy) < min_periods:
                continue
            r, p = stats.pearsonr(xy.iloc[:, 0], xy.iloc[:, 1])
            if np.isnan(r):
                continue
            if abs(r) >= threshold and p <= p_value_cutoff:
                G.add_edge(assets[i], assets[j], weight=round(r, 6), p_value=round(p, 6))

    return G


def build_spearman_graph(
    returns: pd.DataFrame,
    threshold: float = 0.3,
    p_value_cutoff: float = 0.05,
    min_periods: int = 30,
) -> nx.Graph:
    """
    Build a correlation network using Spearman rank correlation.
    Robust to outliers and captures monotonic (non-linear) relationships.
    """
    assets = returns.columns.tolist()
    n = len(assets)
    G = nx.Graph()
    G.add_nodes_from(assets)

    for i in range(n):
        for j in range(i + 1, n):
            xy = pd.concat([returns.iloc[:, i], returns.iloc[:, j]], axis=1).dropna()
            if len(xy) < min_periods:
                continue
            r, p = stats.spearmanr(xy.iloc[:, 0], xy.iloc[:, 1])
            if np.isnan(r):
                continue
            if abs(r) >= threshold and p <= p_value_cutoff:
                G.add_edge(assets[i], assets[j], weight=round(r, 6), p_value=round(p, 6))

    return G


def build_partial_correlation_graph(
    returns: pd.DataFrame,
    threshold: float = 0.15,
    method: str = "glasso",
) -> nx.Graph:
    """
    Build a partial correlation graph (direct dependencies only).

    Removes indirect effects by conditioning on all other variables.
    Uses Graphical Lasso for sparse precision matrix estimation.

    Parameters
    ----------
    returns : pd.DataFrame
        Asset returns matrix.
    threshold : float
        Minimum |partial correlation| to include an edge.
    method : str
        'glasso' (Graphical LASSO, sparse) or 'ledoitwolf' (shrinkage estimator).

    Returns
    -------
    nx.Graph
        Weighted undirected graph. Edge attribute: 'weight' (partial correlation).
    """
    data = returns.dropna().values
    assets = returns.columns.tolist()
    G = nx.Graph()
    G.add_nodes_from(assets)

    try:
        if method == "glasso":
            model = GraphicalLassoCV(cv=5, max_iter=200)
        else:
            model = LedoitWolf()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(data)

        prec = model.precision_  # precision matrix = inverse covariance
        # Partial correlation: rho_ij = -prec_ij / sqrt(prec_ii * prec_jj)
        n = len(assets)
        for i in range(n):
            for j in range(i + 1, n):
                denom = np.sqrt(prec[i, i] * prec[j, j])
                if denom == 0:
                    continue
                pcorr = -prec[i, j] / denom
                if abs(pcorr) >= threshold:
                    G.add_edge(assets[i], assets[j], weight=round(float(pcorr), 6))

    except Exception as e:
        warnings.warn(f"Partial correlation estimation failed: {e}. Falling back to Pearson.")
        return build_pearson_graph(returns, threshold=threshold)

    return G


# ─────────────────────────────────────────────────────────────
# 2. Distance Transform (Mantegna 1999)
# ─────────────────────────────────────────────────────────────

def correlation_to_distance(G: nx.Graph) -> nx.Graph:
    """
    Transform correlation weights to Mantegna distances: d = sqrt(2*(1 - rho)).
    Distances satisfy metric axioms, enabling hierarchical clustering and MST.
    """
    D = G.copy()
    for u, v, data in D.edges(data=True):
        rho = data.get("weight", 0.0)
        # Clamp to [-1, 1] to avoid numerical issues
        rho = max(-1.0, min(1.0, rho))
        D[u][v]["distance"] = round(np.sqrt(2.0 * (1.0 - rho)), 6)
    return D


# ─────────────────────────────────────────────────────────────
# 3. Graph Sparsification
# ─────────────────────────────────────────────────────────────

def minimum_spanning_tree(G: nx.Graph, weight: str = "distance") -> nx.Graph:
    """
    Compute the Minimum Spanning Tree (MST) of the correlation graph.
    MST retains the N-1 most important edges, revealing the backbone.

    Best used after correlation_to_distance() so MST minimises distance
    (= maximises correlation).
    """
    if G.number_of_edges() == 0:
        return G.copy()
    mst = nx.minimum_spanning_tree(G, weight=weight)
    return mst


def planar_maximally_filtered_graph(G: nx.Graph, weight: str = "distance") -> nx.Graph:
    """
    Compute the Planar Maximally Filtered Graph (PMFG).
    PMFG retains 3*(N-2) edges while ensuring planarity.
    Generalisation of MST with richer topology.

    Reference: Tumminello et al. (2005) PNAS 102(30), 10421-10426
    """
    if G.number_of_edges() == 0:
        return G.copy()

    # Sort edges by weight (ascending distance = descending correlation)
    edges_sorted = sorted(G.edges(data=True), key=lambda e: e[2].get(weight, 0))

    pmfg = nx.Graph()
    pmfg.add_nodes_from(G.nodes(data=True))
    max_edges = 3 * (G.number_of_nodes() - 2)

    for u, v, data in edges_sorted:
        pmfg.add_edge(u, v, **data)
        if not nx.is_planar(pmfg):
            pmfg.remove_edge(u, v)
        if pmfg.number_of_edges() >= max_edges:
            break

    return pmfg


def threshold_sparsification(G: nx.Graph, keep_fraction: float = 0.2) -> nx.Graph:
    """
    Keep only the top-k% edges by absolute weight.

    Parameters
    ----------
    keep_fraction : float
        Fraction of edges to retain (0 < keep_fraction <= 1).
    """
    edges = sorted(G.edges(data=True), key=lambda e: abs(e[2].get("weight", 0)), reverse=True)
    k = max(1, int(len(edges) * keep_fraction))
    sparse = nx.Graph()
    sparse.add_nodes_from(G.nodes(data=True))
    for u, v, data in edges[:k]:
        sparse.add_edge(u, v, **data)
    return sparse


# ─────────────────────────────────────────────────────────────
# 4. Spectral Graph Learning
# ─────────────────────────────────────────────────────────────

def spectral_graph_learning(
    returns: pd.DataFrame,
    alpha: float = 1.0,
    beta: float = 0.1,
) -> Tuple[nx.Graph, np.ndarray]:
    """
    Learn the graph Laplacian from data via the GL-SigRep framework.

    Solves:  min_{L} trace(S @ L) + alpha * ||L||_F^2 + beta * ||w||_1
    where S is the empirical covariance and L is the graph Laplacian.

    Approximated here via the precision matrix approach with
    Ledoit-Wolf covariance shrinkage.

    Returns
    -------
    G : nx.Graph
        Learned sparse graph.
    laplacian : np.ndarray
        Graph Laplacian matrix.

    Reference: Dong et al. (2016) "Learning Laplacian Matrix in
    Smooth Graph Signal Representations." IEEE TSP 64(23), 6160-6173.
    """
    data = returns.dropna().values
    assets = returns.columns.tolist()
    n = len(assets)

    lw = LedoitWolf()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lw.fit(data)

    S = lw.covariance_

    # Precision matrix → adjacency of conditional independence graph
    prec = lw.precision_
    # Zero out diagonal; threshold off-diagonal
    W = np.abs(prec.copy())
    np.fill_diagonal(W, 0)

    # Regularize: soft-threshold with beta
    W = np.maximum(W - beta, 0)

    # Build graph Laplacian: L = D - W
    D = np.diag(W.sum(axis=1))
    L = D - W

    # Build networkx graph
    G = nx.Graph()
    G.add_nodes_from(assets)
    for i in range(n):
        for j in range(i + 1, n):
            if W[i, j] > 0:
                G.add_edge(assets[i], assets[j], weight=round(float(W[i, j]), 6))

    return G, L


# ─────────────────────────────────────────────────────────────
# 5. Convenience: Full Pipeline
# ─────────────────────────────────────────────────────────────

def build_correlation_suite(
    returns: pd.DataFrame,
    threshold: float = 0.3,
    sparsify: bool = True,
) -> dict:
    """
    Build a complete suite of correlation graphs for comparative analysis.

    Returns
    -------
    dict with keys:
        'pearson'   : Pearson correlation graph
        'spearman'  : Spearman correlation graph
        'partial'   : Partial correlation graph (Graphical LASSO)
        'mst'       : MST of Pearson distance graph
        'pmfg'      : PMFG of Pearson distance graph
        'spectral'  : Spectral/Laplacian-learned graph
    """
    suite = {}

    print("🔗 Building Pearson correlation graph...")
    suite["pearson"] = build_pearson_graph(returns, threshold=threshold)

    print("🔗 Building Spearman correlation graph...")
    suite["spearman"] = build_spearman_graph(returns, threshold=threshold)

    print("🔗 Building Partial correlation graph (Graphical LASSO)...")
    suite["partial"] = build_partial_correlation_graph(returns, threshold=threshold * 0.5)

    print("🌲 Computing Minimum Spanning Tree (MST)...")
    dist_g = correlation_to_distance(suite["pearson"])
    suite["mst"] = minimum_spanning_tree(dist_g)

    print("✈️  Computing Planar Maximally Filtered Graph (PMFG)...")
    suite["pmfg"] = planar_maximally_filtered_graph(dist_g)

    print("📐 Spectral graph learning...")
    suite["spectral"], _ = spectral_graph_learning(returns)

    for name, g in suite.items():
        print(
            f"   [{name:8s}] nodes={g.number_of_nodes():3d}  "
            f"edges={g.number_of_edges():4d}  "
            f"density={nx.density(g):.4f}"
        )

    return suite


# ─────────────────────────────────────────────────────────────
# Self-test / demo
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    np.random.seed(42)
    n_assets, n_obs = 20, 252
    # Simulate correlated returns
    cov = np.random.uniform(0.2, 0.8, (n_assets, n_assets))
    cov = (cov + cov.T) / 2
    np.fill_diagonal(cov, 1.0)
    cov = cov / np.max(np.abs(np.linalg.eigvals(cov))) * 0.95 + np.eye(n_assets) * 0.05
    returns_sim = pd.DataFrame(
        np.random.multivariate_normal(np.zeros(n_assets), cov, n_obs),
        columns=[f"Asset_{i+1}" for i in range(n_assets)],
    )
    suite = build_correlation_suite(returns_sim)
    print("\n✅ Correlation graph suite built successfully.")
