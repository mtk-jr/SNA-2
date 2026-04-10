import numpy as np
import pandas as pd
import networkx as nx
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ContagionConfig:
    infection_threshold: float = 0.3
    recovery_rate: float = 0.05
    liquidity_penalty: float = 0.2
    max_steps: int = 50
    seed: int = 42


@dataclass
class ContagionResult:
    """Output bundle returned after one simulation run."""
    infected_timeline: List[List[str]]          # which nodes infected at each step
    stress_levels: Dict[str, List[float]]        # stress time-series per node
    final_infected: List[str]                    # nodes that never recovered
    cascade_depth: int                           # how many steps until stabilisation
    systemic_risk_score: float                   # 0-1 aggregate severity
    summary_df: pd.DataFrame                     # tabular summary per stock


class ContagionSimulator:


    def __init__(
        self,
        adjacency_matrix: np.ndarray,
        node_names: List[str],
        config: Optional[ContagionConfig] = None,
    ):
        self.adj = adjacency_matrix
        self.nodes = node_names
        self.n = len(node_names)
        self.config = config or ContagionConfig()
        self.node_idx = {name: i for i, name in enumerate(node_names)}

        # Build a NetworkX graph for analysis utilities
        self.G = nx.from_numpy_array(adjacency_matrix)
        nx.relabel_nodes(self.G, {i: name for i, name in enumerate(node_names)}, copy=False)

        np.random.seed(self.config.seed)
        logger.info(f"ContagionSimulator ready: {self.n} nodes, threshold={self.config.infection_threshold}")

  
    def run(
        self,
        seed_nodes: List[str],
        initial_stress: Optional[Dict[str, float]] = None,
    ) -> ContagionResult:
     
        # --- Initialise state ---
        stress = np.zeros(self.n)
        status = np.zeros(self.n, dtype=int)   # 0=healthy, 1=stressed, 2=recovered

        for node in seed_nodes:
            if node not in self.node_idx:
                raise ValueError(f"Seed node '{node}' not in graph.")
            idx = self.node_idx[node]
            stress[idx] = 1.0
            status[idx] = 1  # immediately stressed

        if initial_stress:
            for node, s in initial_stress.items():
                stress[self.node_idx[node]] = s
                if s > self.config.infection_threshold:
                    status[self.node_idx[node]] = 1

        # Storage
        stress_history = {name: [stress[i]] for i, name in enumerate(self.nodes)}
        infected_timeline: List[List[str]] = [[n for n in seed_nodes]]

        # --- Simulation loop ---
        for step in range(self.config.max_steps):
            new_stress = stress.copy()
            newly_infected = []

            for i in range(self.n):
                if status[i] == 2:
                    continue  # already recovered

                # Weighted stress pressure from neighbours
                neighbour_pressure = 0.0
                total_weight = 0.0
                for j in range(self.n):
                    if i == j:
                        continue
                    w = abs(self.adj[i, j])
                    if w < 1e-6:
                        continue
                    total_weight += w
                    # Defaulted neighbour adds extra liquidity penalty
                    penalty = self.config.liquidity_penalty if status[j] == 1 else 0.0
                    neighbour_pressure += w * (stress[j] + penalty)

                if total_weight > 0:
                    neighbour_pressure /= total_weight

                # Update stress
                if status[i] == 0:  # healthy
                    if neighbour_pressure > self.config.infection_threshold:
                        new_stress[i] = min(1.0, stress[i] + neighbour_pressure * 0.5)
                        status[i] = 1
                        newly_infected.append(self.nodes[i])
                elif status[i] == 1:  # stressed
                    new_stress[i] = min(1.0, stress[i] + neighbour_pressure * 0.3)
                    # Chance of recovery
                    if np.random.random() < self.config.recovery_rate:
                        status[i] = 2
                        new_stress[i] *= 0.5  # partial stress relief

            stress = new_stress
            for i, name in enumerate(self.nodes):
                stress_history[name].append(stress[i])

            infected_timeline.append(newly_infected)
            logger.debug(f"Step {step+1}: {len(newly_infected)} new infections")

            # Early stopping: no new infections and no stressed nodes spreading
            if len(newly_infected) == 0 and not np.any(stress > self.config.infection_threshold):
                logger.info(f"Cascade stabilised at step {step+1}")
                break

        # --- Build result ---
        final_infected = [self.nodes[i] for i in range(self.n) if status[i] == 1]
        systemic_risk = float(np.mean(stress))
        cascade_depth = sum(1 for t in infected_timeline if len(t) > 0)

        summary_df = pd.DataFrame({
            "ticker": self.nodes,
            "final_stress": stress,
            "status": [["healthy", "stressed", "recovered"][s] for s in status],
            "peak_stress": [max(stress_history[n]) for n in self.nodes],
        }).sort_values("final_stress", ascending=False)

        return ContagionResult(
            infected_timeline=infected_timeline,
            stress_levels=stress_history,
            final_infected=final_infected,
            cascade_depth=cascade_depth,
            systemic_risk_score=systemic_risk,
            summary_df=summary_df,
        )

    def run_liquidity_cascade(
        self,
        seed_nodes: List[str],
        liquidity_reserves: Optional[Dict[str, float]] = None,
    ) -> pd.DataFrame:
      
        solvency = np.zeros(self.n)
        liquidity = np.ones(self.n)   # 1 = full liquidity

        reserves = liquidity_reserves or {n: np.random.uniform(0.2, 0.8) for n in self.nodes}

        for node in seed_nodes:
            idx = self.node_idx[node]
            solvency[idx] = 1.0
            liquidity[idx] = 0.0  # defaulted: liquidity gone

        records = []
        for step in range(self.config.max_steps):
            new_solvency = solvency.copy()
            new_liquidity = liquidity.copy()

            for i in range(self.n):
                liq_drain = 0.0
                for j in range(self.n):
                    if i == j:
                        continue
                    w = abs(self.adj[i, j])
                    if solvency[j] > 0.8:  # j is failing → fire sale
                        liq_drain += w * (1 - liquidity[j])

                new_liquidity[i] = max(0, liquidity[i] - liq_drain * 0.3)

                # Liquidity shortfall → solvency pressure
                liq_shortfall = 1 - new_liquidity[i]
                if liq_shortfall > reserves.get(self.nodes[i], 0.3):
                    new_solvency[i] = min(1.0, solvency[i] + liq_shortfall * 0.4)

            solvency, liquidity = new_solvency, new_liquidity
            records.append({
                "step": step,
                "avg_solvency_stress": float(np.mean(solvency)),
                "avg_liquidity": float(np.mean(liquidity)),
                "failed_nodes": int(np.sum(solvency > 0.8)),
            })

            if np.all(solvency < 0.1) or step == self.config.max_steps - 1:
                break

        return pd.DataFrame(records)


if __name__ == "__main__":
    np.random.seed(0)
    tickers = ["AAPL", "MSFT", "JPM", "GS", "XOM", "CVX", "AMZN", "GOOGL"]
    n = len(tickers)

    # Fake adjacency matrix (replace with Member 2's output in production)
    adj = np.random.uniform(0, 1, (n, n))
    adj = (adj + adj.T) / 2
    np.fill_diagonal(adj, 0)

    sim = ContagionSimulator(adj, tickers, ContagionConfig(infection_threshold=0.25))
    result = sim.run(seed_nodes=["JPM", "GS"])

    print("\n=== Contagion Result ===")
    print(f"Systemic Risk Score : {result.systemic_risk_score:.4f}")
    print(f"Cascade Depth       : {result.cascade_depth} steps")
    print(f"Final Infected      : {result.final_infected}")
    print("\nSummary:")
    print(result.summary_df.to_string(index=False))