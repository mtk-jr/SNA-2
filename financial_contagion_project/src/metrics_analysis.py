import networkx as nx
from typing import Dict, Any


# =====================================================================
# 1. NODE-LEVEL METRICS  (CLEAN VERSION – ONLY WHAT YOU USE)
# =====================================================================
def compute_node_metrics(G: nx.DiGraph) -> Dict[str, Dict[str, float]]:
    """
    Computes node-level metrics: degree, betweenness, eigenvector.
    PageRank removed as requested.
    """

    und = G.to_undirected()

    # Degree-related
    deg = dict(und.degree())
    in_deg = dict(G.in_degree())
    out_deg = dict(G.out_degree())
    deg_c = nx.degree_centrality(und)

    # Betweenness
    bet = nx.betweenness_centrality(und)

    # Eigenvector (safe version)
    try:
        eig = nx.eigenvector_centrality(und, max_iter=500)
    except Exception:
        eig = {n: 0.0 for n in G.nodes()}

    # Prepare output
    metrics = {}
    for node in G.nodes():
        metrics[node] = {
            "degree": deg.get(node, 0),
            "in_degree": in_deg.get(node, 0),
            "out_degree": out_deg.get(node, 0),
            "degree_centrality": deg_c.get(node, 0.0),
            "betweenness": bet.get(node, 0.0),
            "eigenvector": eig.get(node, 0.0),
            "capital": G.nodes[node].get("capital", 0.0),
        }

    return metrics



# =====================================================================
# 2. GLOBAL METRICS  (CLEAN VERSION)
# =====================================================================
def compute_global_metrics(G: nx.DiGraph) -> Dict[str, Any]:
    """
    Computes essential global metrics.
    PageRank and extra stats removed.
    """

    if G.number_of_nodes() == 0:
        return {
            "num_nodes": 0,
            "num_edges": 0,
            "density": 0.0,
            "avg_clustering": None,
            "avg_shortest_path_length": None,
        }

    und = G.to_undirected()

    density = nx.density(und)

    # Clustering (safe)
    try:
        avg_clustering = nx.average_clustering(und)
    except Exception:
        avg_clustering = None

    # Shortest path (safe)
    try:
        if nx.is_connected(und):
            avg_path = nx.average_shortest_path_length(und)
        else:
            avg_path = None
    except Exception:
        avg_path = None

    return {
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
        "density": density,
        "avg_clustering": avg_clustering,
        "avg_shortest_path_length": avg_path,
    }
