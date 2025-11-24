import networkx as nx
import pandas as pd
import numpy as np
from typing import Dict, List, Set, Tuple


# ---------------------------------------------------------------------
# 1. Load Network
# ---------------------------------------------------------------------
def load_network_from_csv(path: str) -> nx.DiGraph:
    """
    Load directed interbank network from CSV file.
    Expected columns: lender, borrower, exposure
    """
    df = pd.read_csv(path)
    G = nx.DiGraph()
    for _, row in df.iterrows():
        G.add_edge(row['lender'], row['borrower'], weight=row['exposure'])
    print(f"✅ Loaded network: {G.number_of_nodes()} banks, {G.number_of_edges()} exposures")
    return G


# ---------------------------------------------------------------------
# 2. Compute Network Metrics
# ---------------------------------------------------------------------
def compute_network_metrics(G: nx.DiGraph) -> Dict[str, Dict[str, float]]:
    """
    Compute degree, betweenness, eigenvector, and clustering centralities.
    """
    metrics = {
        'degree': dict(G.degree()),
        'betweenness': nx.betweenness_centrality(G, normalized=True),
        'eigenvector': nx.eigenvector_centrality(G.to_undirected(), max_iter=1000),
        'clustering': nx.clustering(G.to_undirected())
    }
    return metrics


def compute_global_metrics(G: nx.DiGraph) -> Dict[str, float]:
    """
    Compute global structural statistics for the network.
    """
    return {
        "Number of Banks": G.number_of_nodes(),
        "Number of Exposures": G.number_of_edges(),
        "Density": nx.density(G),
        "Average Degree": sum(dict(G.degree()).values()) / G.number_of_nodes(),
        "Average Clustering": nx.average_clustering(G.to_undirected()),
    }


# ---------------------------------------------------------------------
# 3. Display Key Nodes
# ---------------------------------------------------------------------
def print_top_influential_nodes(metrics: Dict[str, Dict[str, float]], top_n: int = 5):
    """
    Display top nodes for each centrality measure.
    """
    for key, vals in metrics.items():
        top_nodes = sorted(vals.items(), key=lambda x: x[1], reverse=True)[:top_n]
        print(f"\n🏦 Top {top_n} banks by {key} centrality:")
        for node, val in top_nodes:
            print(f"  {node:<15}  {val:.4f}")


# ---------------------------------------------------------------------
# 4. Contagion Simulation
# ---------------------------------------------------------------------
def simulate_contagion(
    G: nx.DiGraph,
    initial_failed: Set[str],
    threshold: float = 0.3,
    max_steps: int = 10
) -> List[Set[str]]:
    """
    Simulate contagion process based on threshold failure model.
    A bank fails if proportion of failed lenders exceeds 'threshold'.
    """
    steps = []
    failed = set(initial_failed)
    steps.append(set(failed))

    print(f"\n⚠️ Starting contagion simulation...")
    print(f"   Initial failed banks: {initial_failed}\n")

    for step in range(max_steps):
        new_failures = set()
        for node in G.nodes():
            if node in failed:
                continue

            incoming_edges = G.in_edges(node, data=True)
            total_exposure = sum([d.get('weight', 1.0) for _, _, d in incoming_edges])
            failed_exposure = sum([d.get('weight', 1.0) for src, _, d in incoming_edges if src in failed])

            if total_exposure > 0 and (failed_exposure / total_exposure) >= threshold:
                new_failures.add(node)

        if not new_failures:
            print(f"✅ No new failures at step {step}. Stopping contagion.\n")
            break

        failed.update(new_failures)
        steps.append(set(failed))
        print(f"Step {step+1}: Newly failed banks → {sorted(new_failures)}")

    print(f"\n💥 Total failed banks after contagion: {len(failed)}")
    print(f"🧾 Final failed set: {sorted(failed)}\n")

    return steps


# ---------------------------------------------------------------------
# 5. Generate Demo Network
# ---------------------------------------------------------------------
def generate_demo_network(n_banks: int = 10, seed: int = 42) -> nx.DiGraph:
    """
    Generate synthetic random interbank network for testing.
    """
    np.random.seed(seed)
    G = nx.gnp_random_graph(n_banks, p=0.3, directed=True)
    DG = nx.DiGraph()
    for u, v in G.edges():
        DG.add_edge(f"Bank_{u}", f"Bank_{v}", weight=np.random.uniform(0.1, 1.0))
    return DG


# ---------------------------------------------------------------------
# 6. Main Execution
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # Generate synthetic network
    G = generate_demo_network(8)
    
    # Compute metrics
    metrics = compute_network_metrics(G)
    print_top_influential_nodes(metrics)
    
    # Show global metrics
    globals = compute_global_metrics(G)
    print("\n🌐 Global Network Metrics:")
    for k, v in globals.items():
        print(f"  {k:<20}: {v:.4f}" if isinstance(v, float) else f"  {k:<20}: {v}")
    
    # Run contagion simulation
    initial_failed = {"Bank_2"}
    cascade = simulate_contagion(G, initial_failed, threshold=0.4)
    
    # Save results
    df_metrics = pd.DataFrame(metrics)
    df_metrics.to_csv("network_metrics_summary.csv")
    print("📁 Metrics saved to network_metrics_summary.csv")
