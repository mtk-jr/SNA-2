"""
multilayer_graph.py
===================
Multi-Layer & Multiplex Financial Network

Combines multiple dependency layers (correlation, Granger, MI, etc.)
into a unified multiplex / aggregate representation.

Layer types:
  - Layer-specific analysis (supra-adjacency matrix)
  - Aggregate weighted fusion
  - Eigenvector-centrality–based layer weighting
  - Structural similarity across layers

Advanced features:
  - Multiplex graph modeling (shared node sets, layer-specific edges)
  - Supra-adjacency matrix construction
  - Inter-layer coupling analysis
  - Layer importance ranking

Journal reference:
  Kivela et al. (2014) "Multilayer Networks"
  J. Complex Networks 2(3), 203-271

  Boccaletti et al. (2014) "The Structure and Dynamics of Multilayer Networks"
  Physics Reports 544(1), 1-122
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from scipy.linalg import block_diag


# ─────────────────────────────────────────────────────────────
# 1. Multiplex Graph Container
# ─────────────────────────────────────────────────────────────

class MultiplexGraph:
    """
    A multiplex network where all layers share the same node set
    but have different edge structures.

    This is the standard model for financial networks where the same
    set of assets can be connected by different dependency types.

    Attributes
    ----------
    nodes : list
        Ordered list of node names (shared across layers).
    layers : dict[str, nx.Graph | nx.DiGraph]
        Named graph layers.
    """

    def __init__(self, nodes: Optional[List[str]] = None):
        self.nodes: List[str] = nodes or []
        self.layers: Dict[str, nx.Graph | nx.DiGraph] = {}
        self._node_index: Dict[str, int] = {n: i for i, n in enumerate(self.nodes)}

    # ── Layer management ──────────────────────────────────────

    def add_layer(self, name: str, graph: nx.Graph | nx.DiGraph) -> None:
        """
        Add a named layer. The union of all node sets is tracked.
        """
        new_nodes = [n for n in graph.nodes() if n not in set(self.nodes)]
        self.nodes.extend(new_nodes)
        self._node_index = {n: i for i, n in enumerate(self.nodes)}
        self.layers[name] = graph
        print(f"  ✅ Layer '{name}' added: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges, directed={graph.is_directed()}")

    def remove_layer(self, name: str) -> None:
        if name in self.layers:
            del self.layers[name]

    def layer_names(self) -> List[str]:
        return list(self.layers.keys())

    def n_layers(self) -> int:
        return len(self.layers)

    def n_nodes(self) -> int:
        return len(self.nodes)

    # ── Supra-adjacency matrix ────────────────────────────────

    def supra_adjacency_matrix(
        self,
        inter_layer_coupling: float = 1.0,
        directed: bool = False,
    ) -> np.ndarray:
        """
        Build the supra-adjacency matrix A_supra of size (N*L) × (N*L)
        where N = number of nodes, L = number of layers.

        Block structure:
          Diagonal blocks = intra-layer adjacency matrices
          Off-diagonal blocks = inter-layer coupling (identity * coupling)

        Parameters
        ----------
        inter_layer_coupling : float
            Strength of connections between the same node in different layers.
        directed : bool
            Whether to treat all layers as directed.

        Returns
        -------
        np.ndarray of shape (N*L, N*L)
        """
        N = self.n_nodes()
        L = self.n_layers()

        if N == 0 or L == 0:
            return np.array([])

        layer_list = list(self.layers.values())
        # Intra-layer blocks
        intra_blocks = []
        for g in layer_list:
            A = nx.to_numpy_array(g, nodelist=self.nodes, weight="weight")
            if not directed and g.is_directed():
                A = (A + A.T) / 2  # symmetrize if needed
            intra_blocks.append(A)

        # Block-diagonal = supra-adjacency intra part
        A_supra = block_diag(*intra_blocks)

        # Add inter-layer coupling (identity blocks between adjacent layers)
        coupling_block = inter_layer_coupling * np.eye(N)
        for i in range(L):
            for j in range(L):
                if i != j:
                    row_start = i * N
                    col_start = j * N
                    A_supra[row_start:row_start + N, col_start:col_start + N] = coupling_block

        return A_supra

    def supra_laplacian(self, inter_layer_coupling: float = 1.0) -> np.ndarray:
        """
        Compute the supra-Laplacian matrix: L_supra = D_supra - A_supra.
        Used for diffusion, random-walk, and spectral analysis on multilayer networks.
        """
        A = self.supra_adjacency_matrix(inter_layer_coupling=inter_layer_coupling)
        D = np.diag(A.sum(axis=1))
        return D - A

    # ── Layer adjacency matrices ──────────────────────────────

    def get_layer_adjacency(self, name: str) -> np.ndarray:
        """Return the adjacency matrix for a named layer."""
        g = self.layers[name]
        return nx.to_numpy_array(g, nodelist=self.nodes, weight="weight")

    def get_all_adjacencies(self) -> Dict[str, np.ndarray]:
        """Return all layer adjacency matrices as a dict."""
        return {name: self.get_layer_adjacency(name) for name in self.layers}

    # ── Layer importance ──────────────────────────────────────

    def layer_importance_scores(self) -> Dict[str, float]:
        """
        Rank layers by their spectral radius (largest eigenvalue of adjacency).
        Higher spectral radius ⟹ more influential layer.
        """
        scores = {}
        for name in self.layers:
            A = self.get_layer_adjacency(name)
            try:
                eigvals = np.linalg.eigvals(A)
                scores[name] = float(np.max(np.abs(eigvals)))
            except Exception:
                scores[name] = 0.0
        total = sum(scores.values()) + 1e-9
        return {k: round(v / total, 6) for k, v in scores.items()}

    # ── Summary ───────────────────────────────────────────────

    def summary(self) -> pd.DataFrame:
        """Return a summary DataFrame of all layers."""
        rows = []
        for name, g in self.layers.items():
            rows.append({
                "layer": name,
                "nodes": g.number_of_nodes(),
                "edges": g.number_of_edges(),
                "directed": g.is_directed(),
                "density": round(nx.density(g), 6),
                "avg_degree": round(
                    sum(dict(g.degree()).values()) / max(g.number_of_nodes(), 1), 4
                ),
            })
        return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────
# 2. Multi-Layer Fusion (Aggregate Graph)
# ─────────────────────────────────────────────────────────────

def fuse_layers_weighted(
    multiplex: MultiplexGraph,
    layer_weights: Optional[Dict[str, float]] = None,
    threshold: float = 0.0,
) -> nx.Graph:
    """
    Fuse multiple layers into a single aggregate weighted graph.

    Edge weight in aggregate = Σ (layer_weight_k * edge_weight_k)

    Parameters
    ----------
    multiplex : MultiplexGraph
    layer_weights : dict, optional
        Weight per layer. If None, equal weights are used.
    threshold : float
        Minimum aggregate weight to keep an edge.

    Returns
    -------
    nx.Graph
        Undirected aggregate graph (directed edges are symmetrised).
    """
    names = multiplex.layer_names()
    if not names:
        return nx.Graph()

    if layer_weights is None:
        layer_weights = {n: 1.0 / len(names) for n in names}

    # Normalise weights
    total_w = sum(layer_weights.values())
    lw = {k: v / total_w for k, v in layer_weights.items()}

    N = multiplex.n_nodes()
    A_agg = np.zeros((N, N))

    for name in names:
        if name not in lw:
            continue
        A = multiplex.get_layer_adjacency(name)
        # Symmetrize directed layers
        A_sym = (A + A.T) / 2
        # Normalize within layer
        mx = A_sym.max()
        if mx > 0:
            A_sym = A_sym / mx
        A_agg += lw[name] * A_sym

    # Build graph
    nodes = multiplex.nodes
    G_fused = nx.Graph()
    G_fused.add_nodes_from(nodes)
    for i in range(N):
        for j in range(i + 1, N):
            w = A_agg[i, j]
            if w > threshold:
                G_fused.add_edge(nodes[i], nodes[j], weight=round(float(w), 6))

    print(f"✅ Fused graph: {G_fused.number_of_nodes()} nodes, {G_fused.number_of_edges()} edges")
    return G_fused


def fuse_layers_eigenvalue(
    multiplex: MultiplexGraph,
    threshold: float = 0.0,
) -> nx.Graph:
    """
    Fuse layers weighted by their spectral importance (layer_importance_scores).
    Gives more weight to structurally influential layers.
    """
    importance = multiplex.layer_importance_scores()
    return fuse_layers_weighted(multiplex, layer_weights=importance, threshold=threshold)


def fuse_layers_intersection(
    multiplex: MultiplexGraph,
    min_layer_agreement: int = 2,
) -> nx.Graph:
    """
    Keep only edges that appear in at least `min_layer_agreement` layers.
    Produces a conservative, high-confidence network.
    """
    edge_counts: dict = {}
    edge_weights: dict = {}

    for name, g in multiplex.layers.items():
        g_und = g.to_undirected()
        for u, v, data in g_und.edges(data=True):
            key = tuple(sorted([u, v]))
            edge_counts[key] = edge_counts.get(key, 0) + 1
            edge_weights[key] = edge_weights.get(key, 0.0) + data.get("weight", 1.0)

    G_inter = nx.Graph()
    G_inter.add_nodes_from(multiplex.nodes)
    for (u, v), count in edge_counts.items():
        if count >= min_layer_agreement:
            avg_w = edge_weights[(u, v)] / count
            G_inter.add_edge(u, v, weight=round(float(avg_w), 6), layer_count=count)

    print(f"✅ Intersection graph (min_agreement={min_layer_agreement}): "
          f"{G_inter.number_of_nodes()} nodes, {G_inter.number_of_edges()} edges")
    return G_inter


# ─────────────────────────────────────────────────────────────
# 3. Inter-layer Structural Similarity
# ─────────────────────────────────────────────────────────────

def interlayer_similarity(multiplex: MultiplexGraph) -> pd.DataFrame:
    """
    Compute pairwise structural similarity between layers.
    Uses Jaccard similarity on the edge sets (after undirecting).

    Returns
    -------
    pd.DataFrame
        Symmetric matrix of Jaccard similarities.
    """
    names = multiplex.layer_names()
    n = len(names)
    sim_matrix = np.zeros((n, n))

    edge_sets: Dict[str, set] = {}
    for name, g in multiplex.layers.items():
        g_und = g.to_undirected()
        edge_sets[name] = set(tuple(sorted([u, v])) for u, v in g_und.edges())

    for i, n1 in enumerate(names):
        for j, n2 in enumerate(names):
            if i == j:
                sim_matrix[i, j] = 1.0
            else:
                inter = len(edge_sets[n1] & edge_sets[n2])
                union = len(edge_sets[n1] | edge_sets[n2])
                sim_matrix[i, j] = inter / union if union > 0 else 0.0

    return pd.DataFrame(sim_matrix, index=names, columns=names)


def interlayer_degree_correlation(multiplex: MultiplexGraph) -> pd.DataFrame:
    """
    Compute Pearson correlation of node degrees across layers.
    High correlation = layers agree on which nodes are hubs.
    """
    nodes = multiplex.nodes
    degree_vectors: Dict[str, List[float]] = {}

    for name, g in multiplex.layers.items():
        deg = dict(g.degree())
        degree_vectors[name] = [float(deg.get(n, 0)) for n in nodes]

    df = pd.DataFrame(degree_vectors, index=nodes)
    return df.corr()


# ─────────────────────────────────────────────────────────────
# 4. Convenience Builder
# ─────────────────────────────────────────────────────────────

def build_multiplex_from_suite(layer_suite: Dict[str, nx.Graph]) -> MultiplexGraph:
    """
    Build a MultiplexGraph from a dict of named graphs.

    Parameters
    ----------
    layer_suite : dict
        {'layer_name': nx.Graph or nx.DiGraph, ...}

    Returns
    -------
    MultiplexGraph
    """
    mpx = MultiplexGraph()
    for name, g in layer_suite.items():
        mpx.add_layer(name, g)
    return mpx


# ─────────────────────────────────────────────────────────────
# Self-test / demo
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    np.random.seed(42)
    nodes = [f"Asset_{i+1}" for i in range(10)]

    # Simulate two random layers
    G1 = nx.gnp_random_graph(10, 0.4, seed=1)
    G1 = nx.relabel_nodes(G1, {i: nodes[i] for i in range(10)})
    nx.set_edge_attributes(G1, {e: np.random.uniform(0.3, 1.0) for e in G1.edges()}, "weight")

    G2 = nx.gnp_random_graph(10, 0.3, seed=2, directed=True)
    G2 = nx.relabel_nodes(G2, {i: nodes[i] for i in range(10)})
    nx.set_edge_attributes(G2, {e: np.random.uniform(0.1, 0.8) for e in G2.edges()}, "weight")

    mpx = build_multiplex_from_suite({"pearson": G1, "granger": G2})

    print("\n📋 Multiplex Summary:")
    print(mpx.summary())

    print("\n🔗 Layer importance:", mpx.layer_importance_scores())
    print("\n📐 Supra-adjacency shape:", mpx.supra_adjacency_matrix().shape)
    print("\n🔀 Inter-layer similarity:\n", interlayer_similarity(mpx))

    G_fused = fuse_layers_weighted(mpx)
    G_inter = fuse_layers_intersection(mpx, min_layer_agreement=1)

    print("\n✅ Multiplex graph module demo complete.")
