import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Literal
from enum import Enum
import logging

logger = logging.getLogger(__name__)



class ShockType(str, Enum):
    SECTOR_CRASH      = "sector_crash"
    RATE_HIKE         = "rate_hike"
    LIQUIDITY_FREEZE  = "liquidity_freeze"
    BLACK_SWAN        = "black_swan"
    GEOPOLITICAL      = "geopolitical"


@dataclass
class ShockEvent:
    
    shock_type: ShockType
    magnitude: float              # 0 to 1
    affected_nodes: List[str] = field(default_factory=list)
    step: int = 0
    description: str = ""


@dataclass
class ShockResult:

    portfolio_value_path: np.ndarray   # portfolio value over time
    stock_return_paths: pd.DataFrame   # per-stock return at each step
    max_drawdown: float
    var_95: float                      # Value at Risk at 95%
    cvar_95: float                     # Conditional VaR at 95%
    breach_events: List[Dict]          # when/which stocks breached thresholds
    recovery_time: Optional[int]       # steps to recover to pre-shock value


class ShockSimulator:
   

    def __init__(
        self,
        adjacency_matrix: np.ndarray,
        node_names: List[str],
        portfolio_weights: Optional[np.ndarray] = None,
        n_monte_carlo: int = 1000,
        seed: int = 42,
    ):
        self.adj = adjacency_matrix
        self.nodes = node_names
        self.n = len(node_names)
        self.node_idx = {name: i for i, name in enumerate(node_names)}
        self.n_mc = n_monte_carlo
        np.random.seed(seed)

        if portfolio_weights is not None:
            assert len(portfolio_weights) == self.n
            self.weights = portfolio_weights / portfolio_weights.sum()
        else:
            self.weights = np.ones(self.n) / self.n

        logger.info(f"ShockSimulator ready: {self.n} stocks, {n_monte_carlo} MC paths")


    def _generate_shock_vector(self, event: ShockEvent) -> np.ndarray:
       
        shock = np.zeros(self.n)
        m = event.magnitude

        if event.shock_type == ShockType.SECTOR_CRASH:
            # Direct hit on named stocks + network propagation
            for node in event.affected_nodes:
                if node in self.node_idx:
                    shock[self.node_idx[node]] = -m

            # Propagate through network: neighbouring stocks get partial hit
            for i in range(self.n):
                for j in range(self.n):
                    if abs(shock[j]) > 0:
                        w = abs(self.adj[i, j])
                        shock[i] += -m * w * 0.5

        elif event.shock_type == ShockType.RATE_HIKE:
            # Rate hike hurts all stocks, but growth stocks more
            base_hit = -m * 0.05      # 5% base market drop per unit magnitude
            shock = np.full(self.n, base_hit)
            # High-beta (high-correlation) stocks get hit harder
            betas = np.mean(np.abs(self.adj), axis=1)
            betas = betas / (betas.max() + 1e-8)
            shock -= betas * m * 0.03

        elif event.shock_type == ShockType.LIQUIDITY_FREEZE:
            # Liquidity freeze: all stocks drop, correlated ones more
            corr_sum = np.sum(np.abs(self.adj), axis=1)
            corr_sum /= corr_sum.max() + 1e-8
            shock = -m * (0.03 + corr_sum * 0.05)

        elif event.shock_type == ShockType.BLACK_SWAN:
            # Random fat-tail: a few stocks get massive hits
            n_hit = max(1, int(self.n * 0.3 * m))
            hit_idx = np.random.choice(self.n, n_hit, replace=False)
            shock[hit_idx] = -np.random.uniform(m * 0.1, m * 0.5, n_hit)

        elif event.shock_type == ShockType.GEOPOLITICAL:
            # Energy/commodity spike: energy stocks gain, others lose
            for i, name in enumerate(self.nodes):
                if any(x in name for x in ["XOM", "CVX", "COP", "BP", "SHEL"]):
                    shock[i] = m * 0.08     # energy stocks rally
                else:
                    shock[i] = -m * 0.04   # others suffer

        return shock

   
    def run_scenario(
        self,
        shock_events: List[ShockEvent],
        n_steps: int = 30,
        base_volatility: float = 0.01,
    ) -> ShockResult:
       
        # Initialise return matrix
        returns = np.zeros((n_steps, self.n))
        shock_schedule = {e.step: e for e in shock_events}

        for t in range(n_steps):
            # Base random walk
            noise = np.random.normal(0, base_volatility, self.n)
            # Network correlation: correlated stocks move together
            corr_component = self.adj @ noise * 0.3
            step_return = noise + corr_component

            # Apply shock if scheduled
            if t in shock_schedule:
                event = shock_schedule[t]
                shock_vec = self._generate_shock_vector(event)
                step_return += shock_vec
                logger.info(f"Step {t}: Applied {event.shock_type} shock (mag={event.magnitude})")

            returns[t] = step_return

        # Portfolio value path
        port_returns = returns @ self.weights
        port_value = np.cumprod(1 + port_returns)

        # Risk metrics
        max_dd = self._max_drawdown(port_value)
        var_95  = float(np.percentile(port_returns, 5))
        cvar_95 = float(np.mean(port_returns[port_returns <= var_95]))

        # Breach events: stocks dropping more than 5% in a single step
        breach_events = []
        for t in range(n_steps):
            for i, name in enumerate(self.nodes):
                if returns[t, i] < -0.05:
                    breach_events.append({"step": t, "ticker": name, "return": returns[t, i]})

        # Recovery time
        pre_shock_level = 1.0
        recovery_time = next(
            (t for t in range(n_steps) if port_value[t] >= pre_shock_level * 0.95),
            None
        )

        return ShockResult(
            portfolio_value_path=port_value,
            stock_return_paths=pd.DataFrame(returns, columns=self.nodes),
            max_drawdown=max_dd,
            var_95=var_95,
            cvar_95=cvar_95,
            breach_events=breach_events,
            recovery_time=recovery_time,
        )

    def monte_carlo_shock(
        self,
        shock_event: ShockEvent,
        n_steps: int = 30,
    ) -> pd.DataFrame:
       
        base_shock = self._generate_shock_vector(shock_event)
        results = []

        for sim_i in range(self.n_mc):
            # Each MC path has a slightly different shock magnitude
            magnitude_noise = np.random.normal(1.0, 0.2)
            sim_shock = base_shock * max(0, magnitude_noise)

            # Random walk path
            port_return_path = []
            for t in range(n_steps):
                noise = np.random.normal(0, 0.01, self.n)
                step_return = noise
                if t == shock_event.step:
                    step_return += sim_shock
                port_return_path.append(float((step_return * self.weights).sum()))

            port_value = np.cumprod(1 + np.array(port_return_path))
            results.append({
                "sim": sim_i,
                "final_value": port_value[-1],
                "min_value": port_value.min(),
                "max_drawdown": self._max_drawdown(port_value),
                "total_return": port_value[-1] - 1,
            })

        df = pd.DataFrame(results)
        var_95  = float(np.percentile(df["total_return"], 5))
        cvar_95 = float(df.loc[df["total_return"] <= var_95, "total_return"].mean())

        logger.info(f"Monte Carlo: VaR(95%)={var_95:.4f}, CVaR(95%)={cvar_95:.4f}")
        df.attrs["var_95"] = var_95
        df.attrs["cvar_95"] = cvar_95
        return df

   
    def historical_stress_test(self) -> pd.DataFrame:
     
        crises = [
            {"name": "2008 Financial Crisis",
             "event": ShockEvent(ShockType.SECTOR_CRASH, magnitude=0.9,
                                 affected_nodes=["JPM", "GS", "BAC"], step=0)},
            {"name": "COVID-19 Crash (Mar 2020)",
             "event": ShockEvent(ShockType.BLACK_SWAN, magnitude=0.8, step=0)},
            {"name": "2022 Rate Hike Cycle",
             "event": ShockEvent(ShockType.RATE_HIKE, magnitude=0.7, step=0)},
            {"name": "Russia-Ukraine (Geopolitical)",
             "event": ShockEvent(ShockType.GEOPOLITICAL, magnitude=0.6, step=0)},
        ]

        rows = []
        for crisis in crises:
            result = self.run_scenario([crisis["event"]], n_steps=60)
            rows.append({
                "crisis": crisis["name"],
                "max_drawdown": result.max_drawdown,
                "var_95": result.var_95,
                "cvar_95": result.cvar_95,
                "recovery_steps": result.recovery_time,
                "n_breaches": len(result.breach_events),
            })

        return pd.DataFrame(rows)

    @staticmethod
    def _max_drawdown(value_path: np.ndarray) -> float:
        """Calculate maximum peak-to-trough drawdown."""
        peak = np.maximum.accumulate(value_path)
        drawdown = (value_path - peak) / peak
        return float(drawdown.min())



if __name__ == "__main__":
    np.random.seed(0)
    tickers = ["AAPL", "MSFT", "JPM", "GS", "XOM", "CVX", "AMZN", "GOOGL"]
    n = len(tickers)
    adj = np.random.uniform(0, 0.8, (n, n))
    adj = (adj + adj.T) / 2
    np.fill_diagonal(adj, 0)

    sim = ShockSimulator(adj, tickers, n_monte_carlo=500)

    # Single scenario: banking sector crash on day 5
    events = [
        ShockEvent(ShockType.SECTOR_CRASH, magnitude=0.7,
                   affected_nodes=["JPM", "GS"], step=5, description="Bank run"),
        ShockEvent(ShockType.RATE_HIKE, magnitude=0.4, step=15, description="Emergency rate hike"),
    ]
    result = sim.run_scenario(events, n_steps=40)

    print(f"\n=== Shock Scenario Results ===")
    print(f"Max Drawdown : {result.max_drawdown:.4f}")
    print(f"VaR (95%)    : {result.var_95:.4f}")
    print(f"CVaR (95%)   : {result.cvar_95:.4f}")
    print(f"Breach Events: {len(result.breach_events)}")
    print(f"Recovery Time: {result.recovery_time} steps")

    # Historical stress tests
    print("\n=== Historical Stress Tests ===")
    hist = sim.historical_stress_test()
    print(hist.to_string(index=False))

    # Monte Carlo
    print("\n=== Monte Carlo (500 paths) ===")
    mc = sim.monte_carlo_shock(events[0], n_steps=30)
    print(f"VaR(95%): {mc.attrs['var_95']:.4f}, CVaR(95%): {mc.attrs['cvar_95']:.4f}")
    print(mc.describe().to_string())