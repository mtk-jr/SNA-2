"""
temporal_graph.py
=================
Dynamic & Temporal Financial Network Builder

Builds time-evolving graphs by computing snapshot graphs over
sliding or expanding windows of asset returns.

Features:
  - Sliding-window correlation snapshots
  - Expanding-window (recursive) graph updates
  - Graph change detection (structural break points)
  - Temporal centrality tracking
  - Graph evolution metrics (edge turnover, stability)
  - Node2Vec graph embeddings per snapshot

Advanced:
  - Online / incremental graph updating
  - Anomaly detection on graph structural metrics

Journal reference:
  Tumminello et al. (2010) "Correlation, Hierarchies, and Networks
  in Financial Markets" Journal of Economic Behavior & Organization 75(1)

  Perozzi et al. (2014) "DeepWalk" / Grover & Leskovec (2016) "node2vec"
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Callable, Dict, Generator, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────
# 1. Graph Snapshot Data Structure
# ─────────────────────────────────────────────────────────────

@dataclass
class GraphSnapshot:
    """A single temporal snapshot of a financial dependency graph."""
    timestamp: pd.Timestamp
    graph: nx.Graph
    window_start: pd.Timestamp
    window_end: pd.Timestamp
    metadata: dict = field(default_factory=dict)

    def n_nodes(self) -> int:
        return self.graph.number_of_nodes()

    def n_edges(self) -> int:
        return self.graph.number_of_edges()

    def density(self) -> float:
        return nx.density(self.graph)

    def avg_weight(self) -> float:
        weights = [d.get("weight", 0) for _, _, d in self.graph.edges(data=True)]
        return float(np.mean(weights)) if weights else 0.0


# ─────────────────────────────────────────────────────────────
# 2. Sliding Window Snapshot Generator
# ─────────────────────────────────────────────────────────────

def _pearson_graph_from_returns(
    window_df: pd.DataFrame,
    threshold: float,
) -> nx.Graph:
    """Build a fast Pearson correlation graph from a windowed return DataFrame."""
    corr = window_df.corr(method="pearson")
    assets = corr.columns.tolist()
    G = nx.Graph()
    G.add_nodes_from(assets)
    for i, a in enumerate(assets):
        for j, b in enumerate(assets):
            if j <= i:
                continue
            r = corr.loc[a, b]
            if not np.isnan(r) and abs(r) >= threshold:
                G.add_edge(a, b, weight=round(float(r), 6))
    return G


def build_sliding_window_snapshots(
    returns: pd.DataFrame,
    window: int = 60,
    step: int = 5,
    threshold: float = 0.3,
    graph_builder: Optional[Callable] = None,
    min_obs: int = 30,
) -> List[GraphSnapshot]:
    """
    Generate a sequence of graph snapshots using a sliding window.

    Parameters
    ----------
    returns : pd.DataFrame
        Asset returns (rows = time, columns = assets).
    window : int
        Window size in trading days.
    step : int
        Step size between consecutive windows.
    threshold : float
        Minimum correlation to include an edge.
    graph_builder : callable, optional
        Custom function(window_df, threshold) → nx.Graph.
        Defaults to Pearson correlation.
    min_obs : int
        Minimum observations required to compute a snapshot.

    Returns
    -------
    list of GraphSnapshot
    """
    if graph_builder is None:
        graph_builder = _pearson_graph_from_returns

    snapshots: List[GraphSnapshot] = []
    dates = returns.index
    T = len(dates)

    for start_idx in range(0, T - window + 1, step):
        end_idx = start_idx + window
        window_df = returns.iloc[start_idx:end_idx].dropna()

        if len(window_df) < min_obs:
            continue

        G = graph_builder(window_df, threshold)
        snap = GraphSnapshot(
            timestamp=dates[end_idx - 1],
            graph=G,
            window_start=dates[start_idx],
            window_end=dates[end_idx - 1],
            metadata={"window": window, "step": step},
        )
        snapshots.append(snap)

    print(f"📸 Generated {len(snapshots)} graph snapshots "
          f"(window={window}, step={step}, threshold={threshold})")
    return snapshots


def build_expanding_window_snapshots(
    returns: pd.DataFrame,
    min_window: int = 60,
    step: int = 10,
    threshold: float = 0.3,
) -> List[GraphSnapshot]:
    """
    Generate snapshots using an expanding window (all history up to current time).
    Useful for studying cumulative structural changes.
    """
    snapshots: List[GraphSnapshot] = []
    dates = returns.index
    T = len(dates)

    for end_idx in range(min_window, T + 1, step):
        window_df = returns.iloc[:end_idx].dropna()
        if len(window_df) < min_window:
            continue
        G = _pearson_graph_from_returns(window_df, threshold)
        snap = GraphSnapshot(
            timestamp=dates[end_idx - 1],
            graph=G,
            window_start=dates[0],
            window_end=dates[end_idx - 1],
            metadata={"type": "expanding"},
        )
        snapshots.append(snap)

    print(f"📸 Generated {len(snapshots)} expanding-window snapshots")
    return snapshots


# ─────────────────────────────────────────────────────────────
# 3. Graph Evolution Metrics
# ─────────────────────────────────────────────────────────────

def compute_evolution_metrics(snapshots: List[GraphSnapshot]) -> pd.DataFrame:
    """
    Compute time-series of graph structural metrics across snapshots.

    Returns
    -------
    pd.DataFrame with columns:
        timestamp, n_edges, density, avg_weight, avg_clustering,
        n_components, edge_turnover
    """
    rows = []
    prev_edges: set | None = None

    for snap in snapshots:
        G = snap.graph
        curr_edges = set(tuple(sorted(e)) for e in G.edges())

        if prev_edges is not None and (len(prev_edges) + len(curr_edges)) > 0:
            added = len(curr_edges - prev_edges)
            removed = len(prev_edges - curr_edges)
            turnover = (added + removed) / (len(prev_edges) + len(curr_edges) + 1e-9)
        else:
            turnover = np.nan

        try:
            avg_clust = nx.average_clustering(G)
        except Exception:
            avg_clust = 0.0

        rows.append({
            "timestamp": snap.timestamp,
            "n_edges": snap.n_edges(),
            "density": snap.density(),
            "avg_weight": snap.avg_weight(),
            "avg_clustering": avg_clust,
            "n_components": nx.number_connected_components(G),
            "edge_turnover": turnover,
        })
        prev_edges = curr_edges

    df = pd.DataFrame(rows)
    if not df.empty:
        df.set_index("timestamp", inplace=True)
    return df


def detect_structural_breaks(
    evolution_df: pd.DataFrame,
    column: str = "density",
    z_threshold: float = 2.5,
) -> pd.DatetimeIndex:
    """
    Detect structural breakpoints in graph evolution using z-score thresholding.

    Parameters
    ----------
    evolution_df : pd.DataFrame
        Output of compute_evolution_metrics().
    column : str
        Metric to monitor for breaks.
    z_threshold : float
        Z-score magnitude to flag as a break.

    Returns
    -------
    pd.DatetimeIndex of detected breakpoint timestamps.
    """
    series = evolution_df[column].dropna()
    mu = series.rolling(window=10, min_periods=3).mean()
    sigma = series.rolling(window=10, min_periods=3).std()
    z = (series - mu) / (sigma + 1e-9)
    breaks = z[abs(z) > z_threshold].index
    print(f"⚠️  Detected {len(breaks)} structural breaks in '{column}'")
    return breaks


def temporal_centrality(
    snapshots: List[GraphSnapshot],
    centrality: str = "betweenness",
) -> pd.DataFrame:
    """
    Track node centrality over time across graph snapshots.

    Parameters
    ----------
    centrality : str
        One of 'degree', 'betweenness', 'eigenvector', 'closeness'.

    Returns
    -------
    pd.DataFrame (rows = timestamps, columns = nodes)
    """
    centrality_fn = {
        "degree": lambda G: dict(G.degree()),
        "betweenness": nx.betweenness_centrality,
        "eigenvector": lambda G: nx.eigenvector_centrality(G, max_iter=500),
        "closeness": nx.closeness_centrality,
    }
    fn = centrality_fn.get(centrality, nx.betweenness_centrality)

    records = []
    for snap in snapshots:
        try:
            c = fn(snap.graph)
        except Exception:
            c = {n: 0.0 for n in snap.graph.nodes()}
        c["_timestamp"] = snap.timestamp
        records.append(c)

    df = pd.DataFrame(records).set_index("_timestamp")
    df.index.name = "timestamp"
    return df.fillna(0.0)


# ─────────────────────────────────────────────────────────────
# 4. Online / Incremental Graph Update
# ─────────────────────────────────────────────────────────────

class OnlineGraphUpdater:
    """
    Incrementally updates a financial network as new data arrives.

    Uses an exponentially-weighted covariance estimate to track
    changing asset correlations in real time, avoiding full
    recomputation on each new observation.

    Parameters
    ----------
    assets : list of str
        Asset names.
    alpha : float
        EWM decay factor (0 < alpha < 1). Smaller = longer memory.
    threshold : float
        Minimum |correlation| to include an edge.
    """

    def __init__(self, assets: List[str], alpha: float = 0.06, threshold: float = 0.3):
        self.assets = assets
        self.alpha = alpha
        self.threshold = threshold
        self.n = len(assets)
        self._mu = np.zeros(self.n)
        self._cov = np.eye(self.n) * 0.01
        self._count = 0
        self.current_graph = nx.Graph()
        self.current_graph.add_nodes_from(assets)

    def update(self, new_returns: np.ndarray) -> nx.Graph:
        """
        Update the graph with a new observation vector.

        Parameters
        ----------
        new_returns : np.ndarray of shape (n_assets,)

        Returns
        -------
        Updated nx.Graph
        """
        assert len(new_returns) == self.n
        self._count += 1
        r = new_returns.astype(float)

        # EWM covariance update
        if self._count == 1:
            self._mu = r.copy()
        else:
            diff = r - self._mu
            self._mu = (1 - self.alpha) * self._mu + self.alpha * r
            self._cov = (1 - self.alpha) * self._cov + self.alpha * np.outer(diff, diff)

        # Compute correlation from covariance
        std = np.sqrt(np.diag(self._cov))
        with np.errstate(divide="ignore", invalid="ignore"):
            corr = self._cov / np.outer(std + 1e-9, std + 1e-9)

        # Rebuild graph
        G = nx.Graph()
        G.add_nodes_from(self.assets)
        for i in range(self.n):
            for j in range(i + 1, self.n):
                r_ij = corr[i, j]
                if not np.isnan(r_ij) and abs(r_ij) >= self.threshold:
                    G.add_edge(self.assets[i], self.assets[j], weight=round(float(r_ij), 6))

        self.current_graph = G
        return G

    def get_graph(self) -> nx.Graph:
        return self.current_graph


# ─────────────────────────────────────────────────────────────
# 5. Node2Vec Graph Embedding
# ─────────────────────────────────────────────────────────────

def _random_walk(G: nx.Graph, start: str, length: int, p: float, q: float) -> List[str]:
    """
    Biased random walk for Node2Vec.
    p = return parameter (> 1 → less backtracking)
    q = in-out parameter (> 1 → BFS-like; < 1 → DFS-like)
    """
    walk = [start]
    for _ in range(length - 1):
        curr = walk[-1]
        neighbours = list(G.neighbors(curr))
        if not neighbours:
            break
        if len(walk) == 1:
            walk.append(np.random.choice(neighbours))
        else:
            prev = walk[-2]
            weights = []
            for nb in neighbours:
                if nb == prev:
                    weights.append(1.0 / p)
                elif G.has_edge(prev, nb):
                    weights.append(1.0)
                else:
                    weights.append(1.0 / q)
            w_arr = np.array(weights)
            w_arr = w_arr / w_arr.sum()
            walk.append(np.random.choice(neighbours, p=w_arr))
    return walk


def node2vec_embedding(
    G: nx.Graph,
    dim: int = 32,
    walk_length: int = 30,
    num_walks: int = 10,
    window: int = 5,
    p: float = 1.0,
    q: float = 1.0,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Compute Node2Vec embeddings for all nodes in graph G.

    Implements a lightweight pure-Python Node2Vec without external
    dependencies (no gensim / node2vec package required).
    Uses Word2Vec-style skip-gram approximation via SGD.

    Parameters
    ----------
    G : nx.Graph
        Input graph (undirected or directed).
    dim : int
        Embedding dimensionality.
    walk_length : int
        Length of each random walk.
    num_walks : int
        Number of walks per node.
    window : int
        Context window for skip-gram.
    p, q : float
        Node2Vec return / in-out hyperparameters.
    seed : int
        Random seed.

    Returns
    -------
    pd.DataFrame
        Shape (n_nodes, dim). Index = node names.
    """
    np.random.seed(seed)
    nodes = list(G.nodes())
    if len(nodes) == 0:
        return pd.DataFrame()

    node_idx = {n: i for i, n in enumerate(nodes)}
    N = len(nodes)

    # ── Generate walks ──────────────────────────────────────
    all_walks: List[List[int]] = []
    for _ in range(num_walks):
        shuffled = nodes.copy()
        np.random.shuffle(shuffled)
        for start in shuffled:
            walk = _random_walk(G, start, walk_length, p, q)
            all_walks.append([node_idx[n] for n in walk])

    # ── Skip-gram with negative sampling (SGD) ──────────────
    # Initialise embeddings
    emb = np.random.normal(0, 0.1, (N, dim))
    ctx = np.random.normal(0, 0.1, (N, dim))

    lr = 0.025
    neg_samples = 5

    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -10, 10)))

    for walk in all_walks:
        for pos, center in enumerate(walk):
            # Positive context
            ctx_range = walk[max(0, pos - window): pos] + walk[pos + 1: pos + window + 1]
            for ctx_node in ctx_range:
                # Positive update
                score = sigmoid(emb[center] @ ctx[ctx_node])
                grad = lr * (1 - score)
                emb[center] += grad * ctx[ctx_node]
                ctx[ctx_node] += grad * emb[center]

                # Negative samples
                neg_nodes = np.random.randint(0, N, neg_samples)
                for neg in neg_nodes:
                    if neg == ctx_node:
                        continue
                    score_neg = sigmoid(emb[center] @ ctx[neg])
                    grad_neg = -lr * score_neg
                    emb[center] += grad_neg * ctx[neg]
                    ctx[neg] += grad_neg * emb[center]

    return pd.DataFrame(emb, index=nodes, columns=[f"dim_{i}" for i in range(dim)])


def embed_all_snapshots(
    snapshots: List[GraphSnapshot],
    dim: int = 16,
    walk_length: int = 20,
    num_walks: int = 5,
) -> Dict[pd.Timestamp, pd.DataFrame]:
    """
    Compute Node2Vec embeddings for all graph snapshots.

    Returns
    -------
    dict: {timestamp → embedding DataFrame}
    """
    embeddings = {}
    for snap in snapshots:
        emb = node2vec_embedding(
            snap.graph, dim=dim, walk_length=walk_length, num_walks=num_walks
        )
        embeddings[snap.timestamp] = emb
    print(f"🧬 Computed Node2Vec embeddings for {len(embeddings)} snapshots (dim={dim})")
    return embeddings


# ─────────────────────────────────────────────────────────────
# 6. Graph Anomaly Detection
# ─────────────────────────────────────────────────────────────

def detect_graph_anomalies(
    snapshots: List[GraphSnapshot],
    metric: str = "density",
    window: int = 10,
    z_threshold: float = 2.0,
) -> pd.Series:
    """
    Detect anomalous graph snapshots using rolling z-score on a structural metric.

    Parameters
    ----------
    metric : str
        'density', 'n_edges', 'avg_weight', or 'avg_clustering'.
    window : int
        Rolling window for z-score baseline.
    z_threshold : float
        |z| > threshold → anomaly.

    Returns
    -------
    pd.Series of bool, indexed by timestamp.
    """
    evolution_df = compute_evolution_metrics(snapshots)

    if metric not in evolution_df.columns:
        raise ValueError(f"Unknown metric '{metric}'. Choose from {list(evolution_df.columns)}")

    s = evolution_df[metric].dropna()
    mu = s.rolling(window=window, min_periods=3).mean()
    sigma = s.rolling(window=window, min_periods=3).std()
    z = (s - mu) / (sigma + 1e-9)
    anomalies = z.abs() > z_threshold

    n_anom = anomalies.sum()
    print(f"🚨 Graph anomaly detection: {n_anom} anomalous snapshots detected "
          f"(metric='{metric}', |z| > {z_threshold})")
    return anomalies


# ─────────────────────────────────────────────────────────────
# Self-test / demo
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    np.random.seed(42)
    dates = pd.date_range("2020-01-02", periods=252, freq="B")
    assets = [f"Asset_{i+1}" for i in range(10)]
    returns_sim = pd.DataFrame(
        np.random.randn(252, 10) * 0.01,
        index=dates,
        columns=assets,
    )

    # Sliding window snapshots
    snapshots = build_sliding_window_snapshots(returns_sim, window=60, step=10)

    # Evolution metrics
    evo = compute_evolution_metrics(snapshots)
    print("\n📊 Evolution metrics (last 3 rows):")
    print(evo.tail(3))

    # Structural breaks
    breaks = detect_structural_breaks(evo, column="density")

    # Temporal centrality
    cent = temporal_centrality(snapshots, centrality="degree")
    print("\n📈 Temporal centrality shape:", cent.shape)

    # Online updater
    updater = OnlineGraphUpdater(assets, alpha=0.05, threshold=0.25)
    for _ in range(20):
        obs = np.random.randn(10) * 0.01
        g = updater.update(obs)
    print(f"\n🔄 Online graph after 20 updates: {g.number_of_edges()} edges")

    # Node2Vec embedding (small graph)
    small_G = snapshots[-1].graph
    emb = node2vec_embedding(small_G, dim=8, walk_length=10, num_walks=3)
    print(f"\n🧬 Node2Vec embedding shape: {emb.shape}")

    # Graph anomaly detection
    anomalies = detect_graph_anomalies(snapshots, metric="density")
    print("\n✅ Temporal graph module demo complete.")
