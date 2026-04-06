"""
graph_engine
============
Member 2 — Financial Graph Architecture Module

Provides a complete toolkit for building, analysing, and evolving
financial dependency networks:

Modules
-------
correlation_graph  : Pearson, Spearman, partial correlation, MST, PMFG, spectral learning
causality_graph    : Granger causality, mutual information, transfer entropy
multilayer_graph   : Multiplex / multi-layer graph fusion and analysis
temporal_graph     : Dynamic snapshots, Node2Vec embeddings, online updating
network_metrics    : Centrality, robustness, DebtRank, community detection

Quick Start
-----------
>>> import pandas as pd
>>> from graph_engine.correlation_graph import build_correlation_suite
>>> from graph_engine.causality_graph import build_causality_suite
>>> from graph_engine.multilayer_graph import build_multiplex_from_suite, fuse_layers_eigenvalue
>>> from graph_engine.temporal_graph import build_sliding_window_snapshots
>>> from graph_engine.network_metrics import full_network_report

>>> # Load returns
>>> returns = pd.read_csv("returns.csv", index_col=0, parse_dates=True)

>>> # Build all graph layers
>>> corr_suite = build_correlation_suite(returns)
>>> caus_suite = build_causality_suite(returns)

>>> # Multiplex fusion
>>> all_layers = {**corr_suite, **caus_suite}
>>> mpx = build_multiplex_from_suite(all_layers)
>>> G_fused = fuse_layers_eigenvalue(mpx)

>>> # Metrics
>>> report = full_network_report(G_fused)
"""

from .correlation_graph import (
    build_pearson_graph,
    build_spearman_graph,
    build_partial_correlation_graph,
    build_correlation_suite,
    minimum_spanning_tree,
    planar_maximally_filtered_graph,
    threshold_sparsification,
    spectral_graph_learning,
    correlation_to_distance,
)

from .causality_graph import (
    build_granger_graph,
    build_mutual_information_graph,
    build_transfer_entropy_graph,
    build_causality_suite,
)

from .multilayer_graph import (
    MultiplexGraph,
    build_multiplex_from_suite,
    fuse_layers_weighted,
    fuse_layers_eigenvalue,
    fuse_layers_intersection,
    interlayer_similarity,
    interlayer_degree_correlation,
)

from .temporal_graph import (
    GraphSnapshot,
    build_sliding_window_snapshots,
    build_expanding_window_snapshots,
    compute_evolution_metrics,
    detect_structural_breaks,
    temporal_centrality,
    OnlineGraphUpdater,
    node2vec_embedding,
    embed_all_snapshots,
    detect_graph_anomalies,
)

from .network_metrics import (
    compute_all_centralities,
    identify_systemically_important_nodes,
    compute_global_metrics,
    detect_communities_louvain,
    detect_communities_spectral,
    modularity_score,
    robustness_attack_simulation,
    network_fragility_index,
    debtrank,
    contagion_index,
    spectral_analysis,
    full_network_report,
)

__all__ = [
    # correlation_graph
    "build_pearson_graph", "build_spearman_graph", "build_partial_correlation_graph",
    "build_correlation_suite", "minimum_spanning_tree", "planar_maximally_filtered_graph",
    "threshold_sparsification", "spectral_graph_learning", "correlation_to_distance",
    # causality_graph
    "build_granger_graph", "build_mutual_information_graph", "build_transfer_entropy_graph",
    "build_causality_suite",
    # multilayer_graph
    "MultiplexGraph", "build_multiplex_from_suite", "fuse_layers_weighted",
    "fuse_layers_eigenvalue", "fuse_layers_intersection",
    "interlayer_similarity", "interlayer_degree_correlation",
    # temporal_graph
    "GraphSnapshot", "build_sliding_window_snapshots", "build_expanding_window_snapshots",
    "compute_evolution_metrics", "detect_structural_breaks", "temporal_centrality",
    "OnlineGraphUpdater", "node2vec_embedding", "embed_all_snapshots", "detect_graph_anomalies",
    # network_metrics
    "compute_all_centralities", "identify_systemically_important_nodes",
    "compute_global_metrics", "detect_communities_louvain", "detect_communities_spectral",
    "modularity_score", "robustness_attack_simulation", "network_fragility_index",
    "debtrank", "contagion_index", "spectral_analysis", "full_network_report",
]
