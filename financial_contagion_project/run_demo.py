#!/usr/bin/env python3
"""
run_demo.py
Main driver for contagion simulation + optional GNN risk training/inference.
"""

import os
import sys
import argparse
import random
import pprint

# Standard data/network libs
import networkx as nx
import pandas as pd

# Allow imports from src/
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

# Project modules (user-provided)
from build_network import load_network_from_csv
from metrics_analysis import compute_node_metrics, compute_global_metrics
from contagion_model import simulate_contagion, balance_sheet_cascade
from visualization import (
    plot_clustered_static,
    plot_clustered_interactive,
    plot_failure_over_time
)

# Try to import GNN helpers (these should exist in src/)
try:
    from gnn_dataset import build_node_feature_matrix, graph_to_edge_index
    from gnn_model import GCNNet, GraphSAGENet, GATNet
    GNN_AVAILABLE = True
except Exception as e:
    print(f"GNN helper import failed: {e}")
    GNN_AVAILABLE = False

# Torch imports (optional)
try:
    import torch
    import torch.nn.functional as F
    from torch_geometric.data import Data
    TORCH_AVAILABLE = True
except Exception as e:
    print(f"PyTorch / PyG import failed: {e}")
    TORCH_AVAILABLE = False

# ------------------ Utilities ------------------ #
def parse_attack_flag(s):
    if not s:
        return None, 0
    s = s.strip()
    if ":" in s:
        m, k = s.split(":", 1)
        try:
            return m.strip().lower(), int(k)
        except Exception:
            return m.strip().lower(), None
    return s.strip().lower(), None


def pick_top_k(G, metric, k):
    if k is None or k <= 0:
        return []
    try:
        if metric == "degree":
            c = nx.degree_centrality(G)
        elif metric == "betweenness":
            c = nx.betweenness_centrality(G)
        elif metric == "eigenvector":
            c = nx.eigenvector_centrality_numpy(G.to_undirected())
        else:
            return []
    except Exception:
        c = {}
    ranked = sorted(c.items(), key=lambda x: x[1], reverse=True)
    return [n for n, _ in ranked[:k]]


def resolve_initial_failures(G, args):
    nodes = list(G.nodes())

    if args.attack:
        m, k = parse_attack_flag(args.attack)
        m = (m or "").strip().lower()

        if m in ("degree", "top_degree"):
            chosen = pick_top_k(G, "degree", k)
            print(f"Attack Strategy = top_degree:{k} → Initial Failures: {chosen}")
            return chosen
        elif m in ("betweenness", "top_betweenness"):
            chosen = pick_top_k(G, "betweenness", k)
            print(f"Attack Strategy = top_betweenness:{k} → Initial Failures: {chosen}")
            return chosen
        elif m in ("eigenvector", "top_eigen"):
            chosen = pick_top_k(G, "eigenvector", k)
            print(f"Attack Strategy = top_eigen:{k} → Initial Failures: {chosen}")
            return chosen
        elif m == "random":
            chosen = random.sample(nodes, min(k or 1, len(nodes)))
            print(f"Attack Strategy = random:{k} → Initial Failures: {chosen}")
            return chosen

        norm_map = {n.lower(): n for n in nodes}
        requested_parts = [part.strip().lower() for part in m.split(",") if part.strip()]
        chosen = [norm_map[p] for p in requested_parts if p in norm_map]

        if chosen:
            print(f"Attack Strategy = node_ids ({args.attack}) → Initial Failures: {chosen}")
        else:
            print(f"Attack Strategy = {args.attack} → No matching nodes found → Initial Failures: []")

        return chosen

    if args.initial:
        valid = [n for n in args.initial if n in nodes]
        if valid:
            print(f"Using manually chosen failures: {valid}")
            return valid

    fallback = [random.choice(nodes)]
    print(f"No failures provided → Using fallback: {fallback}")
    return fallback


# ------------------ GNN helpers ------------------ #
def safe_tensor(x):
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch not available")
    import numpy as np
    if isinstance(x, torch.Tensor):
        return x.float()
    if isinstance(x, (list, tuple)):
        x = np.array(x, dtype=float)
    if isinstance(x, (np.ndarray,)):
        return torch.from_numpy(x).float()
    return torch.tensor(x, dtype=torch.float)


def _parse_build_node_feature_matrix_output(out):

    if not isinstance(out, tuple):
        raise ValueError("build_node_feature_matrix must return a tuple (X, node_map, index_map)")

    if len(out) < 2:
        raise ValueError("build_node_feature_matrix must return at least (X, node_map)")

    X = out[0]
    map2 = out[1]
    map3 = out[2] if len(out) >= 3 else None

    if isinstance(map2, dict):
        node_to_idx = map2
        if isinstance(map3, dict):
            idx_to_node = map3
            node_list = [idx_to_node[i] for i in range(len(idx_to_node))]
        else:
            idx_to_node = {idx: n for n, idx in node_to_idx.items()}
            node_list = [idx_to_node[i] for i in range(len(idx_to_node))]
    else:
        if isinstance(map2, (list, tuple)):
            node_list = list(map2)
            node_to_idx = {n: i for i, n in enumerate(node_list)}
            idx_to_node = {i: n for i, n in enumerate(node_list)}
        else:
            raise ValueError("Unexpected node mapping returned by build_node_feature_matrix")

    return X, node_list, node_to_idx, idx_to_node


# ------------------ GNN training & inference ------------------ #
def run_gnn_training(G: nx.DiGraph, failed_nodes, model_name: str = "sage",
                     epochs: int = 200, lr: float = 5e-3, hidden: int = 64):

    if not (GNN_AVAILABLE and TORCH_AVAILABLE):
        print("GNN or PyTorch not available. Skipping GNN training.")
        return None

    print(f"\nTraining GNN Model: {model_name.upper()}")

    try:
        out = build_node_feature_matrix(G)
    except Exception as e:
        print(f"build_node_feature_matrix failed: {e}")
        return None

    try:
        X_raw, node_list, node_to_idx, idx_to_node = _parse_build_node_feature_matrix_output(out)
    except Exception as e:
        print(f"Failed to normalize build_node_feature_matrix output: {e}")
        return None

    try:
        x = safe_tensor(X_raw)
    except Exception:
        import numpy as np
        x = torch.from_numpy(np.array(X_raw, dtype=float)).float()

    try:
        eout = graph_to_edge_index(G, node_list)
    except Exception as e:
        print(f"graph_to_edge_index failed: {e}")
        return None

    if isinstance(eout, torch.Tensor):
        edge_index = eout.long()
    else:
        import numpy as np
        arr = np.array(eout)
        if arr.ndim == 2 and arr.shape[0] == 2:
            edge_index = torch.from_numpy(arr).long()
        elif arr.ndim == 2 and arr.shape[1] == 2:
            edge_index = torch.from_numpy(arr.T).long()
        else:
            raise ValueError("graph_to_edge_index produced unsupported shape")

    y = torch.zeros(x.shape[0], dtype=torch.long)
    for n in failed_nodes:
        if n in node_to_idx:
            y[node_to_idx[n]] = 1

    in_dim = x.shape[1]
    out_dim = 2

    if model_name.lower() == "gcn":
        model = GCNNet(in_dim, hidden, out_dim)
    elif model_name.lower() == "gat":
        model = GATNet(in_dim, hidden, out_dim)
    else:
        model = GraphSAGENet(in_dim, hidden, out_dim)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss()

    model.train()
    print("Starting GNN training...")
    for epoch in range(epochs):
        optimizer.zero_grad()
        out_logits = model(x, edge_index)
        if out_logits.dim() == 1:
            out_logits = out_logits.unsqueeze(1)
            out_logits = torch.cat([-out_logits, out_logits], dim=1)
        loss = loss_fn(out_logits, y)
        loss.backward()
        optimizer.step()
        if epoch % 20 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch} | Loss = {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        logits = model(x, edge_index)
        if logits.dim() == 1:
            logits = logits.unsqueeze(1)
            logits = torch.cat([-logits, logits], dim=1)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

    gnn_prob = {node_list[i]: float(probs[i]) for i in range(len(node_list))}

    out_df = pd.DataFrame.from_dict(gnn_prob, orient="index", columns=["failure_prob"])
    out_df.index.name = "node"
    out_df.to_csv("gnn_node_risks.csv")
    print("GNN node risk saved → gnn_node_risks.csv")

    return gnn_prob


# ------------------ MAIN ------------------ #
def main():
    parser = argparse.ArgumentParser(description="Financial Contagion + optional GNN")
    parser.add_argument("--csv", type=str, help="Path to network CSV (source,target,exposure)")
    parser.add_argument("--initial", nargs="+", help="Initial failed node names")
    parser.add_argument(
        "--attack",
        type=str,
        help="Attack: random:K, top_degree:K, top_betweenness:K, top_eigen:K, "
             "or one/many node ids like Bank25 or Bank25,Bank30"
    )
    parser.add_argument("--threshold", type=float, default=0.4, help="Failure threshold")
    parser.add_argument("--gnn_model", type=str, help="If provided, trains small GNN: sage | gcn | gat")
    args = parser.parse_args()

    default_csv = os.path.join(os.path.dirname(__file__), "data", "sample_interbank_network.csv")
    csv_path = args.csv if args.csv else default_csv

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    print(f"Loading network: {csv_path}")
    G = load_network_from_csv(csv_path)
    print(f"Loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    print("\nBEFORE Contagion:")
    global_metrics = compute_global_metrics(G)
    for k, v in global_metrics.items():
        print(f"  {k}: {v}")

    initial_failures = resolve_initial_failures(G, args)

    print(f"\nRunning contagion (threshold={args.threshold})...")
    sim_result = simulate_contagion(G, initial_failures, threshold=args.threshold)
    failed_nodes = set(sim_result.get("failed_nodes", []))
    history = sim_result.get("history", [])

    print(f"Total Failed: {len(failed_nodes)} / {G.number_of_nodes()}")

    survivors = [n for n in G.nodes() if n not in failed_nodes]
    if len(survivors) == 0:
        print("\nNetwork collapsed entirely → no post metrics.")
    else:
        print("\nAFTER Contagion:")
        post_metrics = compute_global_metrics(G.subgraph(survivors))
        for k, v in post_metrics.items():
            print(f"  {k}: {v}")

    print("\nRunning balance-sheet cascade...")
    cascade = balance_sheet_cascade(G, initial_failures)
    print(f"Cascade steps: {len(cascade.get('history', []))}")
    print(f"Cascade failures: {len(cascade.get('failed_nodes', []))}")

    gnn_probs = None
    if args.gnn_model:
        model_name = args.gnn_model.strip().lower()
        if not GNN_AVAILABLE or not TORCH_AVAILABLE:
            print("Cannot run GNN: required modules missing. Skipping.")
        else:
            gnn_probs = run_gnn_training(G, failed_nodes, model_name)

    print("\nGenerating visualizations...")
    try:
        plot_clustered_static(G, failed_nodes, title="Contagion Outcome (Static)")
    except Exception as e:
        print(f"plot_clustered_static failed: {e}")
    try:
        plot_failure_over_time(history)
    except Exception as e:
        print(f"plot_failure_over_time failed: {e}")
    try:
        fig = plot_clustered_interactive(G, failed=cascade.get("failed_nodes", set()), title="Cascade (Interactive)")
        fig.show()
    except Exception as e:
        print(f"plot_clustered_interactive failed: {e}")

    print("\nSaving node metrics...")
    node_metrics = compute_node_metrics(G)
    df = pd.DataFrame(node_metrics).T
    if gnn_probs:
        df["gnn_failure_prob"] = df.index.map(lambda n: gnn_probs.get(n, None))
    df.to_csv("network_metrics_summary.csv")
    print("Saved → network_metrics_summary.csv")
    print("Saving file to:", os.getcwd())


    print("\nDone.")


if __name__ == "__main__":
    main()
