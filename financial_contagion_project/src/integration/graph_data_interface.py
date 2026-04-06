"""
graph_data_interface.py
=======================
Integration Layer: Data Pipeline → Graph Engine

This module bridges Member 1 (Data Engineer) and Member 2 (Graph Architect).
It accepts standardised feature DataFrames from the data pipeline and
produces graph objects ready for Member 3 (AI/GNN) and Member 4 (Decision).

Input Contract (from Member 1):
  - returns DataFrame : pd.DataFrame(index=DatetimeIndex, columns=asset_names)
  - features DataFrame: pd.DataFrame with technical indicators, sentiment, macros

Output Contract (for Member 3 & 4):
  - nx.Graph / nx.DiGraph objects
  - Adjacency matrices as np.ndarray or torch.Tensor
  - Node feature matrices aligned with graph node ordering
  - Graph snapshots with timestamps

Usage
-----
>>> from integration.graph_data_interface import GraphDataInterface
>>> gdi = GraphDataInterface(returns_df, features_df)
>>> graphs = gdi.build_all()
>>> adj = gdi.get_adjacency("fused")
>>> node_features = gdi.get_node_features("fused")
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd

# Member 2 graph engine imports
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graph_engine.correlation_graph import build_correlation_suite
from graph_engine.causality_graph import build_causality_suite
from graph_engine.multilayer_graph import (
    build_multiplex_from_suite,
    fuse_layers_eigenvalue,
    fuse_layers_intersection,
)
from graph_engine.temporal_graph import (
    build_sliding_window_snapshots,
    compute_evolution_metrics,
)
from graph_engine.network_metrics import (
    compute_all_centralities,
    full_network_report,
)


class GraphDataInterface:
    """
    Central interface object for converting return data into graph representations.

    Parameters
    ----------
    returns : pd.DataFrame
        Asset return series. Index must be DatetimeIndex.
    features : pd.DataFrame, optional
        Node-level feature matrix from Member 1's feature engineering pipeline.
        Index = asset names (columns aligned with returns.columns).
    config : dict, optional
        Override default graph construction parameters.
    """

    DEFAULTS = {
        "corr_threshold": 0.3,
        "granger_alpha": 0.05,
        "mi_threshold": 0.1,
        "te_threshold": 0.05,
        "window": 60,
        "step": 10,
        "min_layer_agreement": 2,
        "fusion_threshold": 0.1,
    }

    def __init__(
        self,
        returns: pd.DataFrame,
        features: Optional[pd.DataFrame] = None,
        config: Optional[dict] = None,
    ):
        self.returns = returns.copy()
        self.features = features
        self.cfg = {**self.DEFAULTS, **(config or {})}

        self._graphs: Dict[str, nx.Graph] = {}
        self._snapshots = None
        self._centrality_cache: Dict[str, pd.DataFrame] = {}

    # ── Build graph suite ─────────────────────────────────────

    def build_all(self, verbose: bool = True) -> Dict[str, nx.Graph]:
        """
        Build the complete suite of financial graphs.

        Returns
        -------
        dict:
            'pearson', 'spearman', 'partial', 'mst', 'pmfg', 'spectral' — correlation layers
            'granger', 'mutual_info', 'transfer_entropy'                 — causality layers
            'fused_eigen'      — eigenvalue-weighted fusion
            'fused_intersect'  — intersection (consensus) graph
        """
        if verbose:
            print("🏗️  Building financial graph suite...")

        # Correlation graphs
        corr_suite = build_correlation_suite(
            self.returns,
            threshold=self.cfg["corr_threshold"],
        )
        self._graphs.update(corr_suite)

        # Causality graphs
        caus_suite = build_causality_suite(
            self.returns,
            granger_alpha=self.cfg["granger_alpha"],
            mi_threshold=self.cfg["mi_threshold"],
            te_threshold=self.cfg["te_threshold"],
        )
        self._graphs.update(caus_suite)

        # Multiplex fusion
        all_layers = {**corr_suite, **caus_suite}
        mpx = build_multiplex_from_suite(all_layers)

        self._graphs["fused_eigen"] = fuse_layers_eigenvalue(
            mpx, threshold=self.cfg["fusion_threshold"]
        )
        self._graphs["fused_intersect"] = fuse_layers_intersection(
            mpx, min_layer_agreement=self.cfg["min_layer_agreement"]
        )

        if verbose:
            print(f"\n✅ Built {len(self._graphs)} graph layers:")
            for name, g in self._graphs.items():
                print(f"   {name:<20} nodes={g.number_of_nodes():3d}  edges={g.number_of_edges():4d}")

        return self._graphs

    def build_temporal(
        self,
        window: Optional[int] = None,
        step: Optional[int] = None,
    ):
        """Build sliding-window graph snapshots for temporal analysis."""
        w = window or self.cfg["window"]
        s = step or self.cfg["step"]
        self._snapshots = build_sliding_window_snapshots(
            self.returns, window=w, step=s,
            threshold=self.cfg["corr_threshold"],
        )
        return self._snapshots

    # ── Adjacency matrix export ───────────────────────────────

    def get_adjacency(
        self,
        graph_name: str = "fused_eigen",
        nodelist: Optional[List[str]] = None,
        as_tensor: bool = False,
    ) -> np.ndarray:
        """
        Get the adjacency matrix for a named graph layer.

        Parameters
        ----------
        graph_name : str
            Name of graph layer (must exist in self._graphs).
        nodelist : list, optional
            Fixed node order. Defaults to sorted asset names.
        as_tensor : bool
            If True, return torch.FloatTensor (requires PyTorch).

        Returns
        -------
        np.ndarray of shape (N, N)
        """
        if graph_name not in self._graphs:
            raise KeyError(f"Graph '{graph_name}' not found. Call build_all() first.")

        G = self._graphs[graph_name]
        nodes = nodelist or sorted(G.nodes())
        A = nx.to_numpy_array(G, nodelist=nodes, weight="weight")

        if as_tensor:
            try:
                import torch
                return torch.FloatTensor(A)
            except ImportError:
                raise ImportError("PyTorch required for as_tensor=True")

        return A

    def get_edge_index(self, graph_name: str = "fused_eigen") -> np.ndarray:
        """
        Get edge index in COO format for PyG / DGL compatibility.

        Returns
        -------
        np.ndarray of shape (2, E) — [source_indices, target_indices]
        """
        if graph_name not in self._graphs:
            raise KeyError(f"Graph '{graph_name}' not found.")

        G = self._graphs[graph_name]
        nodes = sorted(G.nodes())
        node_idx = {n: i for i, n in enumerate(nodes)}

        src, dst = [], []
        for u, v in G.edges():
            src.append(node_idx[u])
            dst.append(node_idx[v])
            if not G.is_directed():
                src.append(node_idx[v])
                dst.append(node_idx[u])

        return np.array([src, dst], dtype=np.int64)

    def get_edge_weights(self, graph_name: str = "fused_eigen") -> np.ndarray:
        """
        Get edge weights as 1-D array aligned with get_edge_index().
        """
        if graph_name not in self._graphs:
            raise KeyError(f"Graph '{graph_name}' not found.")

        G = self._graphs[graph_name]
        weights = []
        for u, v, data in G.edges(data=True):
            w = data.get("weight", 1.0)
            weights.append(w)
            if not G.is_directed():
                weights.append(w)

        return np.array(weights, dtype=np.float32)

    # ── Node features ─────────────────────────────────────────

    def get_node_features(
        self,
        graph_name: str = "fused_eigen",
        include_centrality: bool = True,
        include_external: bool = True,
    ) -> pd.DataFrame:
        """
        Build a node feature matrix aligned with graph node ordering.

        Combines:
          - Network centrality features (betweenness, pagerank, etc.)
          - External features from Member 1's pipeline (if provided)

        Parameters
        ----------
        include_centrality : bool
            Include graph-derived centrality features.
        include_external : bool
            Include Member 1's pipeline features (requires self.features).

        Returns
        -------
        pd.DataFrame (index = nodes, columns = feature names)
        """
        if graph_name not in self._graphs:
            raise KeyError(f"Graph '{graph_name}' not found.")

        G = self._graphs[graph_name]
        nodes = sorted(G.nodes())
        feature_parts = []

        if include_centrality:
            if graph_name not in self._centrality_cache:
                self._centrality_cache[graph_name] = compute_all_centralities(G)
            cent_df = self._centrality_cache[graph_name]
            # Keep only numeric centrality columns
            numeric_cols = cent_df.select_dtypes(include=np.number).columns
            feature_parts.append(cent_df[numeric_cols].reindex(nodes).fillna(0.0))

        if include_external and self.features is not None:
            # Align external features to graph nodes
            ext = self.features.reindex(nodes).fillna(0.0)
            feature_parts.append(ext)

        if not feature_parts:
            # Fall back to degree as minimal feature
            deg = pd.Series(dict(G.degree()), name="degree")
            return pd.DataFrame(deg).reindex(nodes).fillna(0.0)

        return pd.concat(feature_parts, axis=1).fillna(0.0)

    def get_node_feature_matrix(
        self,
        graph_name: str = "fused_eigen",
        as_tensor: bool = False,
    ) -> np.ndarray:
        """
        Return node features as np.ndarray or torch.FloatTensor.
        Shape: (N, F) where N = nodes, F = feature dimension.
        """
        df = self.get_node_features(graph_name)
        arr = df.values.astype(np.float32)

        if as_tensor:
            try:
                import torch
                return torch.FloatTensor(arr)
            except ImportError:
                raise ImportError("PyTorch required for as_tensor=True")

        return arr

    # ── Summary ───────────────────────────────────────────────

    def graph_summary(self) -> pd.DataFrame:
        """Return a summary table of all built graphs."""
        rows = []
        for name, G in self._graphs.items():
            rows.append({
                "layer": name,
                "nodes": G.number_of_nodes(),
                "edges": G.number_of_edges(),
                "directed": G.is_directed(),
                "density": round(nx.density(G), 6),
            })
        return pd.DataFrame(rows).set_index("layer")

    def get_graph(self, name: str) -> nx.Graph:
        """Retrieve a named graph layer."""
        return self._graphs[name]

    def get_snapshots(self):
        """Return temporal snapshots (build_temporal() must be called first)."""
        if self._snapshots is None:
            raise RuntimeError("Call build_temporal() first.")
        return self._snapshots


# ─────────────────────────────────────────────────────────────
# Quick-start factory function
# ─────────────────────────────────────────────────────────────

def build_graph_interface(
    returns: pd.DataFrame,
    features: Optional[pd.DataFrame] = None,
    config: Optional[dict] = None,
    build_temporal: bool = False,
) -> GraphDataInterface:
    """
    One-call factory: build all graphs and return the interface object.

    Parameters
    ----------
    returns : pd.DataFrame
        Asset returns.
    features : pd.DataFrame, optional
        Node features from Member 1.
    config : dict, optional
        Configuration overrides.
    build_temporal : bool
        Also build sliding-window temporal snapshots.

    Returns
    -------
    GraphDataInterface (graphs already built)
    """
    gdi = GraphDataInterface(returns, features=features, config=config)
    gdi.build_all()
    if build_temporal:
        gdi.build_temporal()
    return gdi


# ─────────────────────────────────────────────────────────────
# Self-test / demo
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    np.random.seed(42)
    dates = pd.date_range("2021-01-04", periods=300, freq="B")
    assets = [f"Stock_{chr(65+i)}" for i in range(12)]
    returns_sim = pd.DataFrame(
        np.random.randn(300, 12) * 0.01,
        index=dates,
        columns=assets,
    )
    # Simulate some Member 1 features
    features_sim = pd.DataFrame(
        np.random.randn(12, 5),
        index=assets,
        columns=["RSI", "MACD", "Sentiment", "PE_ratio", "Momentum"],
    )

    gdi = build_graph_interface(returns_sim, features=features_sim)

    print("\n📋 Graph summary:")
    print(gdi.graph_summary())

    print("\n🔢 Fused adjacency matrix shape:", gdi.get_adjacency("fused_eigen").shape)
    print("🔢 Edge index shape:", gdi.get_edge_index("fused_eigen").shape)
    print("🔢 Node feature matrix shape:", gdi.get_node_feature_matrix("fused_eigen").shape)

    print("\n✅ Graph-data interface demo complete.")
