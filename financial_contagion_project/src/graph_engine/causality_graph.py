"""
causality_graph.py
==================
Directional Dependency Networks for Financial Time Series

Builds DIRECTED graphs from:
  1. Granger Causality  — linear predictive causality (VAR-based)
  2. Mutual Information — non-linear / non-parametric dependency
  3. Transfer Entropy   — directional information flow

Advanced features:
  - Lag-order selection via AIC/BIC
  - FDR correction for multiple hypothesis testing
  - Symbolic Transfer Entropy (fast approximation)

Journal references:
  Granger (1969) "Investigating Causal Relations by Econometric Models"
  Econometrica 37(3), 424-438

  Schreiber (2000) "Measuring Information Transfer"
  PRL 85(2), 461-464
"""

from __future__ import annotations

import warnings
from itertools import permutations
from typing import Dict, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import chi2, f as f_dist
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
from statsmodels.tsa.stattools import adfuller, grangercausalitytests


# ─────────────────────────────────────────────────────────────
# 1. Granger Causality Network
# ─────────────────────────────────────────────────────────────

def _adf_stationarity(series: pd.Series, max_diff: int = 2) -> pd.Series:
    """Auto-difference a series until it is ADF-stationary (p < 0.05)."""
    s = series.dropna().copy()
    for _ in range(max_diff):
        result = adfuller(s, autolag="AIC")
        if result[1] < 0.05:
            return s
        s = s.diff().dropna()
    return s


def _select_lag_order(x: np.ndarray, y: np.ndarray, max_lag: int = 10) -> int:
    """Select VAR lag order via BIC to avoid over-fitting."""
    best_bic = np.inf
    best_lag = 1
    n = len(y)
    for lag in range(1, min(max_lag + 1, n // 5)):
        # Build lagged design matrix
        X_lags = np.column_stack([np.roll(x, l)[lag:] for l in range(1, lag + 1)] +
                                 [np.roll(y, l)[lag:] for l in range(1, lag + 1)])
        Y = y[lag:]
        try:
            mdl = OLS(Y, add_constant(X_lags)).fit(disp=0)
            k = X_lags.shape[1] + 1
            bic = n * np.log(mdl.ssr / n) + k * np.log(n)
            if bic < best_bic:
                best_bic = bic
                best_lag = lag
        except Exception:
            continue
    return best_lag


def build_granger_graph(
    returns: pd.DataFrame,
    max_lag: int = 5,
    alpha: float = 0.05,
    fdr_correction: bool = True,
    stationarize: bool = True,
    min_periods: int = 60,
) -> nx.DiGraph:
    """
    Build a directed Granger causality network.

    An edge X → Y exists if lagged values of X Granger-cause Y
    (i.e., past X improves prediction of Y beyond Y's own history).

    Parameters
    ----------
    returns : pd.DataFrame
        Return series (rows = time, columns = assets).
    max_lag : int
        Maximum lag order to test.
    alpha : float
        Significance level (after FDR correction if enabled).
    fdr_correction : bool
        Apply Benjamini-Hochberg FDR correction for multiple tests.
    stationarize : bool
        Auto-difference non-stationary series.

    Returns
    -------
    nx.DiGraph
        Edge X → Y with attributes: 'weight' (F-stat), 'p_value', 'lag'.
    """
    assets = returns.columns.tolist()
    G = nx.DiGraph()
    G.add_nodes_from(assets)

    # Collect all p-values for FDR correction
    raw_results: list[tuple] = []  # (i, j, f_stat, p_val, lag)

    series_dict: dict[str, pd.Series] = {}
    for col in assets:
        s = returns[col].dropna()
        if stationarize:
            s = _adf_stationarity(s)
        series_dict[col] = s

    pairs = list(permutations(range(len(assets)), 2))

    for (i, j) in pairs:
        xi, xj = assets[i], assets[j]
        # Align series
        df_pair = pd.concat([series_dict[xi], series_dict[xj]], axis=1).dropna()
        if len(df_pair) < min_periods:
            continue
        x_vals = df_pair.iloc[:, 0].values
        y_vals = df_pair.iloc[:, 1].values

        # Select lag
        lag = _select_lag_order(x_vals, y_vals, max_lag=max_lag)

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gc_result = grangercausalitytests(df_pair.values, maxlag=lag, verbose=False)
            # Use F-test at selected lag
            f_stat = gc_result[lag][0]["ssr_ftest"][0]
            p_val = gc_result[lag][0]["ssr_ftest"][1]
            raw_results.append((xi, xj, f_stat, p_val, lag))
        except Exception:
            continue

    # FDR correction (Benjamini-Hochberg)
    if fdr_correction and raw_results:
        p_vals = np.array([r[3] for r in raw_results])
        m = len(p_vals)
        order = np.argsort(p_vals)
        p_adj = np.empty(m)
        for rank, idx in enumerate(order):
            p_adj[idx] = min(1.0, p_vals[idx] * m / (rank + 1))
        # Use adjusted p-values
        adjusted = [(r[0], r[1], r[2], p_adj[k], r[4]) for k, r in enumerate(raw_results)]
    else:
        adjusted = [(r[0], r[1], r[2], r[3], r[4]) for r in raw_results]

    for xi, xj, f_stat, p_val, lag in adjusted:
        if p_val <= alpha:
            G.add_edge(
                xi, xj,
                weight=round(float(f_stat), 6),
                p_value=round(float(p_val), 6),
                lag=lag,
            )

    print(f"✅ Granger causality graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} directed edges")
    return G


# ─────────────────────────────────────────────────────────────
# 2. Mutual Information Network
# ─────────────────────────────────────────────────────────────

def _mutual_information_knn(
    x: np.ndarray,
    y: np.ndarray,
    k: int = 5,
) -> float:
    """
    Estimate mutual information via k-nearest-neighbour (KNN) approach.
    Kraskov et al. (2004) estimator: I(X;Y) ≈ psi(k) - <psi(n_x)> - <psi(n_y)> + psi(N)

    Parameters
    ----------
    k : int
        Number of nearest neighbours.
    """
    from scipy.special import digamma
    from scipy.spatial import cKDTree

    n = len(x)
    xy = np.column_stack([x, y])
    tree_xy = cKDTree(xy)
    tree_x = cKDTree(x.reshape(-1, 1))
    tree_y = cKDTree(y.reshape(-1, 1))

    # Find k-th neighbour distances in joint space (Chebyshev = max norm)
    dists, _ = tree_xy.query(xy, k=k + 1, p=np.inf)
    eps = dists[:, -1]  # k-th neighbour distance

    # Count points within eps in marginals
    nx_counts = np.array([len(tree_x.query_ball_point([[xi]], r=e, p=np.inf)) - 1
                           for xi, e in zip(x, eps)])
    ny_counts = np.array([len(tree_y.query_ball_point([[yi]], r=e, p=np.inf)) - 1
                           for yi, e in zip(y, eps)])

    mi = (digamma(k)
          - np.mean(digamma(np.maximum(nx_counts, 1)))
          - np.mean(digamma(np.maximum(ny_counts, 1)))
          + digamma(n))
    return max(0.0, float(mi))


def build_mutual_information_graph(
    returns: pd.DataFrame,
    threshold: float = 0.1,
    method: str = "knn",
    bins: int = 10,
    min_periods: int = 60,
) -> nx.Graph:
    """
    Build an undirected Mutual Information (MI) dependency network.

    MI captures both linear and non-linear dependencies, making it
    more general than Pearson correlation.

    Parameters
    ----------
    returns : pd.DataFrame
        Asset return series.
    threshold : float
        Minimum MI value to include an edge (in nats).
    method : str
        'knn' for KNN estimator (continuous) or 'histogram' (discrete approx).
    bins : int
        Number of bins for histogram MI (only used when method='histogram').

    Returns
    -------
    nx.Graph
        Undirected graph with edge attribute 'weight' = mutual information.
    """
    assets = returns.columns.tolist()
    n = len(assets)
    G = nx.Graph()
    G.add_nodes_from(assets)

    # Normalise to [0,1] for stable MI estimation
    data = returns.dropna()

    for i in range(n):
        for j in range(i + 1, n):
            xy = pd.concat([data.iloc[:, i], data.iloc[:, j]], axis=1).dropna()
            if len(xy) < min_periods:
                continue

            x = xy.iloc[:, 0].values.astype(float)
            y = xy.iloc[:, 1].values.astype(float)

            # Standardise
            x = (x - x.mean()) / (x.std() + 1e-9)
            y = (y - y.mean()) / (y.std() + 1e-9)

            try:
                if method == "knn":
                    mi = _mutual_information_knn(x, y)
                else:
                    # Histogram method
                    c_xy, _, _ = np.histogram2d(x, y, bins=bins)
                    c_x = c_xy.sum(axis=1)
                    c_y = c_xy.sum(axis=0)
                    n_total = c_xy.sum()
                    with np.errstate(divide="ignore", invalid="ignore"):
                        p_xy = c_xy / n_total
                        p_x = c_x / n_total
                        p_y = c_y / n_total
                        outer = np.outer(p_x, p_y)
                        mask = (p_xy > 0) & (outer > 0)
                        mi = float(np.sum(p_xy[mask] * np.log(p_xy[mask] / outer[mask])))

                if mi >= threshold:
                    G.add_edge(assets[i], assets[j], weight=round(mi, 6))
            except Exception:
                continue

    print(f"✅ Mutual information graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


# ─────────────────────────────────────────────────────────────
# 3. Transfer Entropy Network
# ─────────────────────────────────────────────────────────────

def _symbolic_transfer_entropy(
    x: np.ndarray,
    y: np.ndarray,
    order: int = 3,
    lag: int = 1,
) -> float:
    """
    Compute Symbolic Transfer Entropy (STE) from X → Y.

    Replaces raw values with ordinal patterns (symbols), giving a
    fast, distribution-free estimate of directed information flow.

    Reference: Staniek & Lehnertz (2008) PRL 100, 158101
    """
    n = len(x)
    from math import factorial
    from itertools import permutations as iperms

    # Map ordinal patterns to integer indices
    patterns = {p: i for i, p in enumerate(iperms(range(order)))}
    n_sym = factorial(order)

    def symbolise(arr: np.ndarray) -> np.ndarray:
        symbols = []
        for t in range(len(arr) - order + 1):
            pat = tuple(np.argsort(arr[t: t + order]))
            symbols.append(patterns.get(pat, 0))
        return np.array(symbols)

    sx = symbolise(x)
    sy = symbolise(y)

    # Align
    min_len = min(len(sx), len(sy)) - lag
    if min_len < 10:
        return 0.0

    sy_future = sy[lag: min_len + lag]
    sy_past = sy[:min_len]
    sx_past = sx[:min_len]

    # Estimate probability distributions using integer counts
    def joint_prob(a, b):
        pairs = list(zip(a, b))
        counts: dict = {}
        for p in pairs:
            counts[p] = counts.get(p, 0) + 1
        total = len(pairs)
        return {k: v / total for k, v in counts.items()}

    def triple_prob(a, b, c):
        triples = list(zip(a, b, c))
        counts: dict = {}
        for t in triples:
            counts[t] = counts.get(t, 0) + 1
        total = len(triples)
        return {k: v / total for k, v in counts.items()}

    p_yf_yp = joint_prob(sy_future, sy_past)
    p_yf_yp_xp = triple_prob(sy_future, sy_past, sx_past)
    p_yp_xp = joint_prob(sy_past, sx_past)
    p_yp = {k[0]: v for k, v in {k: sum(vv for kk, vv in
             joint_prob(sy_past, sx_past).items() if kk[0] == k[0])
             for k in joint_prob(sy_past, sx_past)}.items()}

    te = 0.0
    for (yf, yp, xp), p_triple in p_yf_yp_xp.items():
        p2 = p_yf_yp.get((yf, yp), 1e-12)
        p3 = p_yp_xp.get((yp, xp), 1e-12)
        p1 = p_yp.get(yp, 1e-12)
        if p_triple > 0 and p2 > 0 and p3 > 0 and p1 > 0:
            te += p_triple * np.log((p_triple * p1) / (p2 * p3))

    return max(0.0, float(te))


def build_transfer_entropy_graph(
    returns: pd.DataFrame,
    threshold: float = 0.05,
    order: int = 3,
    lag: int = 1,
    min_periods: int = 100,
) -> nx.DiGraph:
    """
    Build a directed Transfer Entropy network.

    An edge X → Y exists if TE(X→Y) > threshold, indicating
    directed information flow from X to Y.

    Parameters
    ----------
    returns : pd.DataFrame
        Asset return series.
    threshold : float
        Minimum TE (in nats) to add a directed edge.
    order : int
        Ordinal pattern length for symbolic TE.
    lag : int
        Time lag for TE computation.

    Returns
    -------
    nx.DiGraph
        Directed graph with edge attribute 'weight' = transfer entropy.
    """
    assets = returns.columns.tolist()
    G = nx.DiGraph()
    G.add_nodes_from(assets)
    data = returns.dropna()

    pairs = list(permutations(range(len(assets)), 2))

    for (i, j) in pairs:
        xi, xj = assets[i], assets[j]
        xy = pd.concat([data[xi], data[xj]], axis=1).dropna()
        if len(xy) < min_periods:
            continue
        x = xy.iloc[:, 0].values.astype(float)
        y = xy.iloc[:, 1].values.astype(float)
        te_xy = _symbolic_transfer_entropy(x, y, order=order, lag=lag)
        if te_xy >= threshold:
            G.add_edge(xi, xj, weight=round(te_xy, 6), lag=lag)

    print(f"✅ Transfer entropy graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} directed edges")
    return G


# ─────────────────────────────────────────────────────────────
# 4. Convenience: Build All Causality Graphs
# ─────────────────────────────────────────────────────────────

def build_causality_suite(
    returns: pd.DataFrame,
    granger_alpha: float = 0.05,
    mi_threshold: float = 0.1,
    te_threshold: float = 0.05,
) -> dict:
    """
    Build the full causality graph suite.

    Returns
    -------
    dict with keys:
        'granger' : Granger causality DiGraph
        'mutual_info' : Mutual information Graph (undirected)
        'transfer_entropy' : Transfer entropy DiGraph
    """
    print("📊 Building causality graph suite...")
    suite = {}
    suite["granger"] = build_granger_graph(returns, alpha=granger_alpha)
    suite["mutual_info"] = build_mutual_information_graph(returns, threshold=mi_threshold)
    suite["transfer_entropy"] = build_transfer_entropy_graph(returns, threshold=te_threshold)
    return suite


# ─────────────────────────────────────────────────────────────
# Self-test / demo
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    np.random.seed(0)
    T, N = 300, 8
    eps = np.random.randn(T, N)
    # Introduce Granger causality: asset 0 leads asset 1
    for t in range(2, T):
        eps[t, 1] += 0.5 * eps[t - 1, 0]
        eps[t, 2] += 0.4 * eps[t - 1, 1]
    returns_sim = pd.DataFrame(eps, columns=[f"Asset_{i+1}" for i in range(N)])

    suite = build_causality_suite(returns_sim)
    print("\nGranger edges:", list(suite["granger"].edges())[:5])
    print("MI edges:", list(suite["mutual_info"].edges())[:5])
    print("TE edges:", list(suite["transfer_entropy"].edges())[:5])
    print("✅ Causality graph suite built successfully.")
