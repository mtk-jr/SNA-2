import torch
import numpy as np
import networkx as nx
from torch_geometric.data import Data

# -----------------------------------------------------------
# 1. Build Node Feature Matrix
# -----------------------------------------------------------
def build_node_feature_matrix(G: nx.Graph):
    """
    Returns:
      X : np.array (num_nodes × num_features)
      node_list : list of node names in correct index order
      node_to_idx : dict {node → index}
    """

    # Fixed ordering of nodes
    node_list = list(G.nodes())
    node_to_idx = {n: i for i, n in enumerate(node_list)}

    # Features
    in_deg = dict(G.in_degree())
    out_deg = dict(G.out_degree())
    deg = dict(G.degree())
    pr = nx.pagerank(G)
    bet = nx.betweenness_centrality(G)

    features = []
    for n in node_list:
        features.append([
            in_deg.get(n, 0),
            out_deg.get(n, 0),
            deg.get(n, 0),
            G.nodes[n].get("capital", 0.0),
            pr.get(n, 0.0),
            bet.get(n, 0.0),
        ])

    X = np.array(features, dtype=np.float32)

    # *** RETURN node_list instead of idx_to_node ***
    return X, node_list, node_to_idx


# -----------------------------------------------------------
# 2. Convert Graph → Edge Index (PyG format)
# -----------------------------------------------------------
def graph_to_edge_index(G: nx.DiGraph, node_list):
    """
    Convert edges using node_to_idx mapping.
    node_list must be list of nodes in index order.
    """

    node_to_idx = {n: i for i, n in enumerate(node_list)}

    edges = []
    for u, v in G.edges():
        edges.append([node_to_idx[u], node_to_idx[v]])

    edges = np.array(edges, dtype=np.int64)

    return edges.T   # [2, E]


# -----------------------------------------------------------
# 3. Build PyG Data object
# -----------------------------------------------------------
def build_pyg_graph(G: nx.DiGraph):
    X, node_list, node_to_idx = build_node_feature_matrix(G)
    edge_index = graph_to_edge_index(G, node_list)

    data = Data(
        x=torch.tensor(X, dtype=torch.float32),
        edge_index=torch.tensor(edge_index, dtype=torch.long)
    )

    data.node_list = node_list
    data.node_to_idx = node_to_idx

    return data
