import copy
import random
import networkx as nx
from typing import List, Dict, Any


# ---------------------------------------------------------------------
# 🧠 Weighted Exposure-Based Contagion Model
# ---------------------------------------------------------------------
def simulate_contagion(
    G: nx.DiGraph,
    initial_failures: List[str],
    threshold: float = 0.4,
    capital_buffer: float = 1.0,
    p_fail: float = None
) -> Dict[str, Any]:
    """
    Simulates contagion in a financial network using weighted exposures.
    A bank fails if (exposure from failed banks / total exposure) > threshold * capital_buffer.
    
    Returns:
        {
            "failed_nodes": set of failed banks,
            "history": list of sets (banks failed per iteration)
        }
    """
    G = copy.deepcopy(G)
    failed = set(initial_failures)
    new_failures = set(initial_failures)
    history = [set(new_failures)]

    while new_failures:
        next_failures = set()

        for node in G.nodes():
            if node not in failed:
                total_exposure = 0.0
                failed_exposure = 0.0

                # Exposure = edge weight (can be named 'weight' or 'exposure')
                for neighbor in G.neighbors(node):
                    exposure = G[node][neighbor].get('weight', G[node][neighbor].get('exposure', 1.0))
                    total_exposure += exposure
                    if neighbor in failed:
                        failed_exposure += exposure

                if total_exposure > 0:
                    frac_failed = failed_exposure / total_exposure

                    # Weighted failure rule
                    if frac_failed >= threshold * capital_buffer:
                        next_failures.add(node)

        # Update failure sets
        new_failures = next_failures - failed
        if not new_failures:
            break

        failed |= new_failures
        history.append(set(new_failures))

    return {
        "failed_nodes": failed,
        "history": history
    }


# ---------------------------------------------------------------------
# 💰 Balance Sheet–Based Contagion Model
# ---------------------------------------------------------------------
def balance_sheet_cascade(G: nx.DiGraph, initial_failures: List[str]) -> Dict[str, Any]:
    """
    Simulates contagion using exposures and capital buffers.
    A bank fails when accumulated losses from failed counterparties exceed its capital.
    
    Returns:
        {
            "failed_nodes": set of failed banks,
            "history": list of sets (banks failed per round),
            "losses": dict of cumulative losses per bank
        }
    """
    G = copy.deepcopy(G)
    failed = set(initial_failures)
    history = [set(failed)]
    losses = {node: 0.0 for node in G.nodes()}
    new_failed = set(failed)

    while new_failed:
        next_failed = set()

        for failed_bank in new_failed:
            for creditor in G.predecessors(failed_bank):
                exposure = G[creditor][failed_bank].get('weight', G[creditor][failed_bank].get('exposure', 0.0))
                losses[creditor] += exposure

                capital = G.nodes[creditor].get('capital', 0.0)

                if losses[creditor] >= capital and creditor not in failed:
                    next_failed.add(creditor)

        new_failed = next_failed - failed
        if not new_failed:
            break

        failed.update(new_failed)
        history.append(set(new_failed))

    return {
        "failed_nodes": failed,
        "history": history,
        "losses": losses
    }
