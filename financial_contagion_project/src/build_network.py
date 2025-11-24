import networkx as nx
import pandas as pd
import random
from typing import Tuple


# ============================================================
# 1. LOAD REAL INTERBANK NETWORK FROM CSV
# ============================================================
def load_network_from_csv(path: str, weighted: bool = True) -> nx.DiGraph:
    """
    Loads a directed interbank network from a CSV file.
    Required columns:
        - source
        - target
        - exposure

    ✔ Multiple rows between same banks -> exposure is summed
    ✔ Missing exposure → defaults to 1.0
    ✔ Each bank gets a capital buffer (10%–30% of its outgoing exposure)
    """

    df = pd.read_csv(path)

    # Validate required columns
    for col in ["source", "target", "exposure"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column in CSV: '{col}'")

    G = nx.DiGraph()

    # -----------------------------
    # Add edges (aggregate weights)
    # -----------------------------
    for _, row in df.iterrows():
        s = str(row["source"])
        t = str(row["target"])

        # exposure may be NaN → set to 1.0
        if weighted:
            try:
                w = float(row["exposure"])
                if pd.isna(w):
                    w = 1.0
            except:
                w = 1.0
        else:
            w = 1.0

        # If edge already exists → add exposure
        if G.has_edge(s, t):
            G[s][t]["exposure"] += w
        else:
            G.add_edge(s, t, exposure=w)

    # -----------------------------
    # Add capital buffers to nodes
    # -----------------------------
    for n in G.nodes():
        outgoing_exposure = sum(G[n][nbr]["exposure"] for nbr in G.successors(n))
        # Add small constant to avoid zero capital
        capital = round(random.uniform(0.10, 0.30) * (outgoing_exposure + 10), 2)
        G.nodes[n]["capital"] = capital

    return G


# ============================================================
# 2. SYNTHETIC NETWORK GENERATOR (Used when no CSV provided)
# ============================================================
def generate_synthetic_network(n_banks: int = 30, prob: float = 0.08, seed: int = 42) -> nx.DiGraph:
    """
    Generates a random but realistic synthetic interbank network.

    ✔ Undirected base → converted to directed
    ✔ Exposure weights are realistic ranges
    ✔ Capital buffers automatically assigned
    """
    random.seed(seed)
    G_und = nx.erdos_renyi_graph(n=n_banks, p=prob, seed=seed)

    # Rename nodes to Bank_1, Bank_2, ...
    mapping = {i: f"Bank_{i+1}" for i in range(n_banks)}
    G_und = nx.relabel_nodes(G_und, mapping)

    G = nx.DiGraph()

    # -----------------------------
    # Add directed exposures
    # -----------------------------
    for u, v in G_und.edges():

        # u → v with 60% chance
        if random.random() < 0.6:
            w = round(random.uniform(0.5, 8.0), 2)
            G.add_edge(u, v, exposure=w)

        # v → u with 40% chance
        if random.random() < 0.4:
            w = round(random.uniform(0.2, 5.0), 2)
            G.add_edge(v, u, exposure=w)

    # Ensure all banks exist
    for i in range(1, n_banks + 1):
        name = f"Bank_{i}"
        if not G.has_node(name):
            G.add_node(name)

    # -----------------------------
    # Add capital buffers
    # -----------------------------
    for n in G.nodes():
        outgoing_exposure = sum(
            G[n][nbr]["exposure"] for nbr in G.successors(n)
        )

        capital = round(random.uniform(0.10, 0.30) * (outgoing_exposure + 10), 2)
        G.nodes[n]["capital"] = capital

    return G
