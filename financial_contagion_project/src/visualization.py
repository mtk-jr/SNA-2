import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from networkx.algorithms.community import greedy_modularity_communities


# ============================================================
# 🔵 1. Compute Communities
# ============================================================
def compute_communities(G):
    und = G.to_undirected()
    communities = greedy_modularity_communities(und)
    node_to_comm = {}

    for i, comm in enumerate(communities):
        for node in comm:
            node_to_comm[node] = i

    return node_to_comm, len(communities)


# ============================================================
# 🔵 2. Shared Layout (Static & Interactive)
# ============================================================
def compute_layout(G):
    """
    Clean layout for reproducibility across all visualizations.
    """
    und = G.to_undirected()
    pos = nx.spring_layout(
        und,
        seed=42,
        k=1.2,
        iterations=200
    )
    return pos


# ============================================================
# 🔵 3. STATIC (Clustered) Visualization
# ============================================================
def plot_clustered_static(G, failed, title="Interbank Network – Failed vs Healthy"):
    failed = set(failed)

    node_to_comm, _ = compute_communities(G)
    pos = compute_layout(G)
    und = G.to_undirected()

    palette = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
    ]

    node_colors = [
        "red" if n in failed else palette[node_to_comm[n] % len(palette)]
        for n in und.nodes()
    ]

    node_sizes = [
        900 if n in failed else 300 for n in und.nodes()
    ]

    plt.figure(figsize=(18, 12))

    nx.draw_networkx_edges(und, pos, width=0.4, alpha=0.4)

    nx.draw_networkx_nodes(
        und,
        pos,
        node_color=node_colors,
        node_size=node_sizes,
        edgecolors="black"
    )

    # Label failed nodes only
    nx.draw_networkx_labels(
        und, pos,
        labels={n: n for n in failed},
        font_color="white",
        font_size=10,
        font_weight="bold"
    )

    plt.title(title, fontsize=20, pad=20)
    plt.axis("off")
    plt.show()


# ============================================================
# 🔵 4. INTERACTIVE Visualization
# ============================================================
def plot_clustered_interactive(G, failed, title="Interbank Network (Interactive)"):
    failed = set(failed)

    node_to_comm, _ = compute_communities(G)
    pos = compute_layout(G)
    und = G.to_undirected()

    # --- Edges ---
    edge_x, edge_y = [], []
    for u, v in und.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        mode="lines",
        line=dict(width=0.5, color="#888"),
        hoverinfo="none"
    )

    # --- Nodes ---
    node_x, node_y = [], []
    node_color, node_size, node_label, node_hover = [], [], [], []

    palette = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
    ]

    for node in und.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

        is_failed = node in failed
        node_color.append("red" if is_failed else palette[node_to_comm[node] % len(palette)])
        node_size.append(22 if is_failed else 12)
        node_label.append(node if is_failed else "")
        node_hover.append(f"{node}<br>Community: {node_to_comm[node]}")

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        text=node_label,
        textposition="top center",
        hovertext=node_hover,
        hoverinfo="text",
        marker=dict(
            size=node_size,
            color=node_color,
            line=dict(width=1, color="black")
        )
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title=title,
        title_x=0.5,
        showlegend=False,
        paper_bgcolor="white",
        plot_bgcolor="white",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(l=0, r=0, t=60, b=0)
    )

    return fig


# ============================================================
# 🔥 5. GNN Heatmap Visualization (NEW)
# ============================================================
def plot_gnn_risk(G, risk_scores, title="GNN Risk Prediction Heatmap"):
    """
    risk_scores: dict {node: risk_value between 0 and 1}
    Colors:
        green → low risk
        yellow → medium risk
        red → high risk
    """

    pos = compute_layout(G)
    und = G.to_undirected()

    # Normalize risks → 0 to 1
    risks = [risk_scores[n] for n in und.nodes()]
    min_r, max_r = min(risks), max(risks)
    norm = lambda r: (r - min_r) / (max_r - min_r + 1e-9)

    node_colors = [
        plt.cm.Reds(norm(risk_scores[n])) for n in und.nodes()
    ]

    plt.figure(figsize=(18, 12))
    nx.draw_networkx_edges(und, pos, width=0.4, alpha=0.4)

    nx.draw_networkx_nodes(
        und, pos,
        node_color=node_colors,
        node_size=[300 for _ in und.nodes()],
        edgecolors="black"
    )

    nx.draw_networkx_labels(
        und, pos,
        font_color="black",
        font_size=8
    )

    plt.title(title, fontsize=20)
    plt.axis("off")
    plt.show()


# ============================================================
# 🔵 6. Failure-Over-Time Plot
# ============================================================
def plot_failure_over_time(history):
    if not history:
        print("⚠ No contagion history.")
        return

    steps = list(range(1, len(history) + 1))
    new_failures = [len(step) for step in history]
    cumulative = [sum(new_failures[:i+1]) for i in range(len(new_failures))]

    plt.figure(figsize=(8, 5))
    plt.plot(steps, cumulative, "-o", color="red", lw=2)
    plt.bar(steps, new_failures, alpha=0.3)

    plt.title("Contagion Spread Over Steps", fontsize=14)
    plt.xlabel("Step")
    plt.ylabel("Number of Failed Banks")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()
