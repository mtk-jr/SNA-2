"""
network_metrics.py
==================
Comprehensive Financial Network Metrics & Robustness Analysis

Covers:
  1. Node-level centrality (degree, betweenness, eigenvector, PageRank,
     Katz, harmonic, percolation, coreness)
  2. Global structural metrics (density, clustering, path length, assortativity)
  3. Community detection (Louvain, Girvan-Newman, spectral)
  4. Network robustness analysis (targeted / random attack simulations)
  5. Systemic risk indicators (DebtRank, contagion index, fragility score)
  6. Spectral analysis (eigenvalue spectrum, spectral gap, Fiedler value)

Journal references:
  Battiston et al. (2012) "DebtRank: Too Central to Fail?"
  Scientific Reports 2, 541

  Albert et al. (2000) "Error and Attack Tolerance of Complex Networks"
  Nature 406, 378-382

  Newman (2010) "Networks: An Introduction" Oxford University Press
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from collections import defaultdict


# ─────────────────────────────────────────────────────────────
# 1. Node Centrality Metrics
# ─────────────────────────────────────────────────────────────

def compute_all_centralities(
    G: nx.Graph,
    weight: str = "weight",
) -> pd.DataFrame:
    """
    Compute a comprehensive suite of node centrality measures.

    Measures computed:
      - degree_centrality
      - in_degree / out_degree (directed only)
      - strength (weighted degree)
      - betweenness_centrality
      - closeness_centrality
      - eigenvector_centrality
      - pagerank
      - katz_centrality
      - harmonic_centrality
      - coreness (k-core number)

    Parameters
    ----------
    G : nx.Graph or nx.DiGraph
    weight : str
        Edge attribute to use as weight.

    Returns
    -------
    pd.DataFrame
        Rows = nodes, columns = centrality metrics.
    """
    nodes = list(G.nodes())
    metrics: Dict[str, Dict] = {}

    # Degree
    metrics["degree"] = dict(G.degree())
    metrics["degree_centrality"] = nx.degree_centrality(G)

    # Directed-only metrics
    if G.is_directed():
        metrics["in_degree"] = dict(G.in_degree())
        metrics["out_degree"] = dict(G.out_degree())
        metrics["in_strength"] = dict(G.in_degree(weight=weight))
        metrics["out_strength"] = dict(G.out_degree(weight=weight))
    else:
        metrics["strength"] = dict(G.degree(weight=weight))

    # Betweenness
    try:
        metrics["betweenness"] = nx.betweenness_centrality(G, weight=weight, normalized=True)
    except Exception:
        metrics["betweenness"] = {n: 0.0 for n in nodes}

    # Closeness
    try:
        metrics["closeness"] = nx.closeness_centrality(G)
    except Exception:
        metrics["closeness"] = {n: 0.0 for n in nodes}

    # Eigenvector (undirected version for stability)
    try:
        G_und = G.to_undirected() if G.is_directed() else G
        metrics["eigenvector"] = nx.eigenvector_centrality_numpy(G_und, weight=weight)
    except Exception:
        metrics["eigenvector"] = {n: 0.0 for n in nodes}

    # PageRank
    try:
        metrics["pagerank"] = nx.pagerank(G, weight=weight, alpha=0.85)
    except Exception:
        metrics["pagerank"] = {n: 1.0 / len(nodes) for n in nodes}

    # Katz centrality
    try:
        lam_max = max(abs(v) for v in nx.adjacency_spectrum(G_und if G.is_directed() else G))
        alpha_katz = 1.0 / (lam_max * 1.1) if lam_max > 0 else 0.1
        metrics["katz"] = nx.katz_centrality_numpy(G, alpha=alpha_katz, weight=weight)
    except Exception:
        metrics["katz"] = {n: 0.0 for n in nodes}

    # Harmonic centrality
    try:
        metrics["harmonic"] = nx.harmonic_centrality(G)
    except Exception:
        metrics["harmonic"] = {n: 0.0 for n in nodes}

    # Core number (k-shell)
    try:
        core_dict = nx.core_number(G.to_undirected() if G.is_directed() else G)
        metrics["coreness"] = core_dict
    except Exception:
        metrics["coreness"] = {n: 0 for n in nodes}

    df = pd.DataFrame(metrics, index=nodes)
    return df.fillna(0.0)


def identify_systemically_important_nodes(
    centrality_df: pd.DataFrame,
    top_n: int = 5,
) -> pd.DataFrame:
    """
    Identify the most systemically important nodes using a composite score.

    Score = normalised sum of: betweenness + eigenvector + pagerank + coreness

    Returns
    -------
    pd.DataFrame of top-n nodes sorted by systemic importance score.
    """
    cols_to_use = [c for c in ["betweenness", "eigenvector", "pagerank", "coreness"]
                   if c in centrality_df.columns]

    if not cols_to_use:
        return centrality_df.head(top_n)

    sub = centrality_df[cols_to_use].copy()
    # Min-max normalise each column
    for col in cols_to_use:
        rng = sub[col].max() - sub[col].min()
        if rng > 0:
            sub[col] = (sub[col] - sub[col].min()) / rng

    centrality_df = centrality_df.copy()
    centrality_df["systemic_importance"] = sub.mean(axis=1)
    return centrality_df.sort_values("systemic_importance", ascending=False).head(top_n)


# ─────────────────────────────────────────────────────────────
# 2. Global Structural Metrics
# ─────────────────────────────────────────────────────────────

def compute_global_metrics(G: nx.Graph) -> Dict[str, float]:
    """
    Compute global network statistics.

    Returns
    -------
    dict with keys:
        n_nodes, n_edges, density, avg_degree, avg_strength,
        avg_clustering, transitivity, diameter (largest component),
        avg_path_length (largest component), assortativity,
        n_components, largest_component_fraction
    """
    N = G.number_of_nodes()
    E = G.number_of_edges()

    result: Dict[str, float] = {
        "n_nodes": N,
        "n_edges": E,
        "density": round(nx.density(G), 6),
    }

    if N == 0:
        return result

    degrees = [d for _, d in G.degree()]
    result["avg_degree"] = round(float(np.mean(degrees)), 4)
    result["max_degree"] = float(max(degrees))
    result["degree_heterogeneity"] = round(float(np.std(degrees) / (np.mean(degrees) + 1e-9)), 4)

    # Weighted strength
    strengths = [d for _, d in G.degree(weight="weight")]
    result["avg_strength"] = round(float(np.mean(strengths)), 4)

    # Clustering
    G_und = G.to_undirected() if G.is_directed() else G
    try:
        result["avg_clustering"] = round(nx.average_clustering(G_und), 6)
        result["transitivity"] = round(nx.transitivity(G_und), 6)
    except Exception:
        result["avg_clustering"] = 0.0
        result["transitivity"] = 0.0

    # Components
    if G.is_directed():
        components = list(nx.weakly_connected_components(G))
    else:
        components = list(nx.connected_components(G))

    result["n_components"] = len(components)
    largest = max(components, key=len)
    result["largest_component_fraction"] = round(len(largest) / N, 4)

    # Diameter & average path length (on largest component)
    try:
        if G.is_directed():
            sub = G.subgraph(largest)
            if nx.is_strongly_connected(sub):
                result["diameter"] = nx.diameter(sub)
                result["avg_path_length"] = round(nx.average_shortest_path_length(sub), 4)
        else:
            sub = G_und.subgraph(largest)
            result["diameter"] = nx.diameter(sub)
            result["avg_path_length"] = round(nx.average_shortest_path_length(sub), 4)
    except Exception:
        result["diameter"] = -1
        result["avg_path_length"] = -1.0

    # Assortativity (degree correlation)
    try:
        result["degree_assortativity"] = round(nx.degree_assortativity_coefficient(G), 4)
    except Exception:
        result["degree_assortativity"] = 0.0

    return result


# ─────────────────────────────────────────────────────────────
# 3. Community Detection
# ─────────────────────────────────────────────────────────────

def detect_communities_louvain(G: nx.Graph) -> Dict[str, int]:
    """
    Detect communities using the Louvain algorithm.
    Falls back to greedy modularity if python-louvain is not installed.

    Returns
    -------
    dict: {node: community_id}
    """
    G_und = G.to_undirected() if G.is_directed() else G
    try:
        import community as community_louvain
        partition = community_louvain.best_partition(G_und)
        n_communities = len(set(partition.values()))
        print(f"🔍 Louvain: {n_communities} communities detected")
        return partition
    except ImportError:
        pass

    # Fallback: greedy modularity
    try:
        communities = nx.community.greedy_modularity_communities(G_und)
        partition = {}
        for cid, community in enumerate(communities):
            for node in community:
                partition[node] = cid
        print(f"🔍 Greedy modularity: {len(communities)} communities detected")
        return partition
    except Exception:
        return {n: 0 for n in G.nodes()}


def detect_communities_spectral(G: nx.Graph, n_clusters: int = 3) -> Dict[str, int]:
    """
    Spectral community detection via k-means on the graph Laplacian eigenvectors.

    Parameters
    ----------
    n_clusters : int
        Number of communities.
    """
    from scipy.linalg import eigh
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import normalize

    G_und = G.to_undirected() if G.is_directed() else G
    nodes = list(G_und.nodes())
    A = nx.to_numpy_array(G_und, weight="weight")
    D = np.diag(A.sum(axis=1))
    L = D - A

    # Normalised Laplacian for better numerical stability
    d_sqrt_inv = np.diag(1.0 / np.sqrt(np.diag(D) + 1e-9))
    L_norm = d_sqrt_inv @ L @ d_sqrt_inv

    eigvals, eigvecs = eigh(L_norm)
    k = min(n_clusters, len(eigvecs))
    X = normalize(eigvecs[:, :k], norm="l2")

    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)

    partition = {nodes[i]: int(labels[i]) for i in range(len(nodes))}
    print(f"🔍 Spectral clustering: {k} communities detected")
    return partition


def modularity_score(G: nx.Graph, partition: Dict[str, int]) -> float:
    """Compute the modularity Q of a given partition."""
    G_und = G.to_undirected() if G.is_directed() else G
    communities = defaultdict(set)
    for node, cid in partition.items():
        communities[cid].add(node)
    try:
        Q = nx.community.modularity(G_und, list(communities.values()))
        return round(Q, 6)
    except Exception:
        return 0.0


# ─────────────────────────────────────────────────────────────
# 4. Robustness Analysis
# ─────────────────────────────────────────────────────────────

def robustness_attack_simulation(
    G: nx.Graph,
    strategy: str = "betweenness",
    fraction: float = 1.0,
) -> pd.DataFrame:
    """
    Simulate network robustness under targeted or random node removal.

    Parameters
    ----------
    strategy : str
        'betweenness'  : Remove highest-betweenness nodes first (targeted)
        'degree'       : Remove highest-degree nodes first (targeted)
        'pagerank'     : Remove highest-pagerank nodes first (targeted)
        'random'       : Random node removal (averaged over 5 trials)
    fraction : float
        Maximum fraction of nodes to remove (0, 1].

    Returns
    -------
    pd.DataFrame with columns:
        fraction_removed, largest_component_fraction, n_edges_remaining
    """
    G_und = G.to_undirected() if G.is_directed() else G
    N = G_und.number_of_nodes()
    max_remove = int(fraction * N)

    # Determine removal order
    if strategy == "random":
        orders = []
        for _ in range(5):
            nodes_shuffled = list(G_und.nodes())
            np.random.shuffle(nodes_shuffled)
            orders.append(nodes_shuffled)
    else:
        centrality_fn = {
            "betweenness": nx.betweenness_centrality,
            "degree": lambda g: dict(g.degree()),
            "pagerank": nx.pagerank,
        }.get(strategy, nx.betweenness_centrality)

        try:
            c = centrality_fn(G_und)
            order = sorted(c.keys(), key=lambda x: c[x], reverse=True)
        except Exception:
            order = list(G_und.nodes())
        orders = [order]

    rows = []
    for order in orders:
        current = G_und.copy()
        for k, node in enumerate(order[:max_remove]):
            if not current.has_node(node):
                continue
            current.remove_node(node)

            if current.number_of_nodes() == 0:
                lcc_frac = 0.0
                e_remain = 0
            else:
                components = list(nx.connected_components(current))
                largest = max(components, key=len)
                lcc_frac = len(largest) / N
                e_remain = current.number_of_edges()

            rows.append({
                "fraction_removed": round((k + 1) / N, 4),
                "largest_component_fraction": round(lcc_frac, 6),
                "n_edges_remaining": e_remain,
                "trial": id(order),
            })

    df = pd.DataFrame(rows)
    # Average over trials if random
    if strategy == "random":
        df = (df.groupby("fraction_removed")
              [["largest_component_fraction", "n_edges_remaining"]]
              .mean()
              .reset_index())
    else:
        df = df.drop(columns=["trial"], errors="ignore")

    return df


def network_fragility_index(G: nx.Graph) -> Dict[str, float]:
    """
    Compute the Network Fragility Index (NFI):
    Area under the LCC-fraction curve under targeted attack.

    Lower NFI = more fragile network (faster LCC collapse).

    Also returns:
      - critical_fraction: node fraction at which LCC drops below 0.5
      - robustness_R: Schneider et al. (2011) robustness measure R = (1/N) Σ lcc(k)
    """
    N = G.number_of_nodes()
    if N <= 2:
        return {"nfi": 1.0, "critical_fraction": 1.0, "robustness_R": 1.0}

    df = robustness_attack_simulation(G, strategy="betweenness", fraction=1.0)
    lcc = df["largest_component_fraction"].values
    frac = df["fraction_removed"].values

    # Area under curve (trapezoidal)
    nfi = float(np.trapz(lcc, frac)) if len(lcc) > 1 else 0.0

    # Critical fraction (LCC < 0.5)
    below_half = frac[lcc < 0.5]
    critical = float(below_half[0]) if len(below_half) > 0 else 1.0

    # Robustness R (Schneider 2011)
    R = float(np.sum(lcc) / N)

    return {
        "nfi": round(nfi, 4),
        "critical_fraction": round(critical, 4),
        "robustness_R": round(R, 4),
    }


# ─────────────────────────────────────────────────────────────
# 5. Systemic Risk Metrics
# ─────────────────────────────────────────────────────────────

def debtrank(
    G: nx.DiGraph,
    initial_shocked: List[str],
    shock_size: float = 0.5,
    max_iter: int = 50,
    tol: float = 1e-6,
) -> Dict[str, float]:
    """
    Compute DebtRank — a recursive measure of systemic impact.

    DebtRank(i) = economic value lost due to node i's distress,
    propagated through the network via balance-sheet exposures.

    Reference: Battiston et al. (2012) Scientific Reports 2, 541

    Parameters
    ----------
    G : nx.DiGraph
        Directed exposure graph. Edge weight = exposure.
    initial_shocked : list
        Nodes to shock initially.
    shock_size : float
        Initial shock magnitude [0, 1] applied to shocked nodes.

    Returns
    -------
    dict: {node: debtrank_score}
    """
    nodes = list(G.nodes())
    n = len(nodes)
    idx = {node: i for i, node in enumerate(nodes)}

    # Build normalised adjacency (row = creditor, col = debtor)
    A = nx.to_numpy_array(G, nodelist=nodes, weight="weight")

    # Normalise rows by total out-exposure (liability side)
    row_sums = A.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        W = np.where(row_sums > 0, A / row_sums, 0.0)

    # Initialise distress levels h ∈ [0, 1]
    h = np.zeros(n)
    for node in initial_shocked:
        if node in idx:
            h[idx[node]] = shock_size

    h_prev = h.copy()
    state = np.zeros(n)  # 0 = susceptible, 1 = distressed, 2 = inactive

    for node in initial_shocked:
        if node in idx:
            state[idx[node]] = 1

    for _ in range(max_iter):
        h_new = h.copy()
        for i in range(n):
            if state[i] == 2:
                continue
            # Impact from distressed neighbours
            impact = W[:, i] @ h_prev  # sum over creditors
            h_new[i] = min(1.0, h[i] + impact)
            if h_new[i] >= 1.0:
                state[i] = 2  # inactivated

        if np.max(np.abs(h_new - h_prev)) < tol:
            break
        h_prev = h.copy()
        h = h_new

    return {nodes[i]: round(float(h[i]), 6) for i in range(n)}


def contagion_index(G: nx.Graph) -> Dict[str, float]:
    """
    Compute the Contagion Index for each node:
    CI(v) = (number of nodes reachable from v if v fails) / N

    A node with high CI is a potential contagion source.
    """
    N = G.number_of_nodes()
    if N == 0:
        return {}

    ci = {}
    for node in G.nodes():
        G_temp = G.copy()
        G_temp.remove_node(node)
        if G_temp.number_of_nodes() == 0:
            ci[node] = 0.0
            continue
        # Reachable nodes from each node in the reduced graph
        if G.is_directed():
            reachable = len(nx.descendants(G, node))
        else:
            # Component size loss
            comp_orig = max(nx.connected_components(G), key=len)
            comp_new = max(nx.connected_components(G_temp), key=len) if G_temp.number_of_nodes() > 0 else set()
            reachable = len(comp_orig) - len(comp_new)
        ci[node] = round(reachable / N, 6)

    return ci


# ─────────────────────────────────────────────────────────────
# 6. Spectral Analysis
# ─────────────────────────────────────────────────────────────

def spectral_analysis(G: nx.Graph) -> Dict[str, float | np.ndarray]:
    """
    Compute spectral properties of the graph Laplacian.

    Returns
    -------
    dict with keys:
        eigenvalues : sorted eigenvalues of normalised Laplacian
        spectral_gap : lambda_2 - lambda_1 (algebraic connectivity)
        spectral_radius : largest eigenvalue
        fiedler_value : second-smallest eigenvalue (algebraic connectivity)
        fiedler_vector : Fiedler vector (for graph partitioning)
    """
    from scipy.linalg import eigh

    G_und = G.to_undirected() if G.is_directed() else G
    nodes = list(G_und.nodes())

    if len(nodes) == 0:
        return {}

    A = nx.to_numpy_array(G_und, weight="weight")
    D = np.diag(A.sum(axis=1))
    L = D - A

    # Normalised Laplacian
    d_sqrt_inv = np.diag(1.0 / np.sqrt(np.diag(D) + 1e-9))
    L_norm = d_sqrt_inv @ L @ d_sqrt_inv

    eigvals, eigvecs = eigh(L_norm)
    eigvals = np.real(eigvals)

    return {
        "eigenvalues": eigvals,
        "spectral_gap": round(float(eigvals[1] - eigvals[0]), 6) if len(eigvals) > 1 else 0.0,
        "spectral_radius": round(float(eigvals[-1]), 6),
        "fiedler_value": round(float(eigvals[1]), 6) if len(eigvals) > 1 else 0.0,
        "fiedler_vector": eigvecs[:, 1] if eigvecs.shape[1] > 1 else eigvecs[:, 0],
    }


# ─────────────────────────────────────────────────────────────
# 7. Full Metrics Report
# ─────────────────────────────────────────────────────────────

def full_network_report(G: nx.Graph, top_n: int = 5) -> dict:
    """
    Generate a comprehensive network metrics report.

    Returns a dict containing:
      'global'       : dict of global metrics
      'centrality'   : pd.DataFrame of node centralities
      'top_nodes'    : pd.DataFrame of top systemically important nodes
      'communities'  : dict of {node: community_id}
      'modularity'   : float
      'robustness'   : dict (nfi, critical_fraction, robustness_R)
      'spectral'     : dict of spectral metrics
    """
    print("📊 Computing full network metrics report...")

    report = {}

    report["global"] = compute_global_metrics(G)
    print(f"  ✅ Global metrics: {report['global']['n_nodes']} nodes, {report['global']['n_edges']} edges")

    report["centrality"] = compute_all_centralities(G)
    report["top_nodes"] = identify_systemically_important_nodes(report["centrality"], top_n=top_n)
    print(f"  ✅ Centrality computed for {len(report['centrality'])} nodes")

    report["communities"] = detect_communities_louvain(G)
    report["modularity"] = modularity_score(G, report["communities"])
    print(f"  ✅ Modularity Q = {report['modularity']:.4f}")

    report["robustness"] = network_fragility_index(G)
    print(f"  ✅ Robustness R = {report['robustness']['robustness_R']:.4f}")

    report["spectral"] = spectral_analysis(G)
    print(f"  ✅ Fiedler value = {report['spectral'].get('fiedler_value', 0):.6f}")

    return report


# ─────────────────────────────────────────────────────────────
# Self-test / demo
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    np.random.seed(42)
    G = nx.barabasi_albert_graph(30, 3, seed=42)
    G = nx.relabel_nodes(G, {i: f"Asset_{i+1}" for i in range(30)})
    for u, v in G.edges():
        G[u][v]["weight"] = np.random.uniform(0.1, 1.0)

    report = full_network_report(G)

    print("\n🏆 Top Systemically Important Nodes:")
    print(report["top_nodes"][["degree", "betweenness", "pagerank", "systemic_importance"]].to_string())

    print("\n🌐 Global Metrics:")
    for k, v in report["global"].items():
        print(f"  {k:<35}: {v}")

    print("\n🔒 Robustness:")
    for k, v in report["robustness"].items():
        print(f"  {k}: {v}")

    # DebtRank test
    DG = nx.DiGraph()
    nodes = [f"Bank_{i+1}" for i in range(8)]
    DG.add_nodes_from(nodes)
    for i in range(7):
        DG.add_edge(nodes[i], nodes[i+1], weight=np.random.uniform(0.3, 1.0))
    dr = debtrank(DG, initial_shocked=["Bank_1"], shock_size=0.6)
    print("\n💣 DebtRank scores:", dr)

    print("\n✅ Network metrics module demo complete.")
