

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
import logging
import decision_engine
logger = logging.getLogger(__name__)


class DecisionInterface:
 

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.adj_matrix: Optional[np.ndarray] = None
        self.node_names: Optional[List[str]] = None
        self.centrality_scores: Optional[pd.DataFrame] = None
        self.crash_probabilities: Optional[Dict[str, float]] = None
        self.predicted_returns: Optional[Dict[str, float]] = None
        self.returns_df: Optional[pd.DataFrame] = None
        self.fundamentals: Optional[Dict[str, Dict]] = None
        self._is_ready = False
        logger.info("DecisionInterface initialised")

    def load_graph_data(
        self,
        adjacency_matrix: np.ndarray,
        node_names: List[str],
        centrality_scores: Optional[pd.DataFrame] = None,
    ) -> "DecisionInterface":
      
        assert adjacency_matrix.shape[0] == adjacency_matrix.shape[1] == len(node_names)
        self.adj_matrix = adjacency_matrix
        self.node_names = node_names
        self.centrality_scores = centrality_scores
        logger.info(f"Graph data loaded: {len(node_names)} nodes")
        return self

    def load_ai_predictions(
        self,
        crash_probabilities: Dict[str, float],
        predicted_returns: Dict[str, float],
    ) -> "DecisionInterface":
   
        self.crash_probabilities = crash_probabilities
        self.predicted_returns = predicted_returns
        logger.info(f"AI predictions loaded: {len(crash_probabilities)} tickers")
        return self

    def load_market_data(
        self,
        returns_df: pd.DataFrame,
        fundamentals: Optional[Dict[str, Dict]] = None,
    ) -> "DecisionInterface":
    
        self.returns_df = returns_df
        self.fundamentals = fundamentals or {}
        logger.info(f"Market data loaded: {len(returns_df)} days, "
                    f"{len(returns_df.columns)} tickers")
        return self

    def run_full_pipeline(
        self,
        seed_nodes: Optional[List[str]] = None,
        shock_magnitude: float = 0.5,
        n_rl_episodes: int = 20,
    ) -> Dict:
       
        self._validate()

        results = {}

        # Step 1: Contagion simulation
        logger.info("Step 1/4: Running contagion simulation...")
        results["contagion_result"] = self._run_contagion(seed_nodes, shock_magnitude)

        # Step 2: Shock simulation
        logger.info("Step 2/4: Running shock simulation...")
        results["shock_result"], results["mc_results"] = self._run_shock(shock_magnitude)

        # Step 3: Investment scoring
        logger.info("Step 3/4: Scoring investments...")
        results["investment_rankings"] = self._run_scoring(
            results["contagion_result"],
            results["shock_result"],
        )

        # Step 4: RL portfolio optimisation
        logger.info("Step 4/4: Running RL portfolio optimisation...")
        if self.returns_df is not None:
            results["portfolio_weights"], results["rl_backtest"] = self._run_rl(n_rl_episodes)
        else:
            # Fallback: CVaR-adjusted weights from scoring
            results["portfolio_weights"] = self._equal_weight_fallback()
            results["rl_backtest"] = pd.DataFrame()

        logger.info("Pipeline complete!")
        return results

    def _run_contagion(self, seed_nodes, shock_magnitude):
        from decision_engine.contagion_market import ContagionSimulator, ContagionConfig

        if seed_nodes is None:
            # Default: pick top-2 betweenness centrality nodes as shock source
            if self.centrality_scores is not None and "betweenness" in self.centrality_scores.columns:
                seed_nodes = (self.centrality_scores
                              .nlargest(2, "betweenness")["ticker"].tolist())
            else:
                seed_nodes = self.node_names[:2]

        cfg = ContagionConfig(
            infection_threshold=0.3 * (1 - shock_magnitude),
            max_steps=50,
        )
        sim = ContagionSimulator(self.adj_matrix, self.node_names, cfg)
        return sim.run(seed_nodes)

    def _run_shock(self, magnitude):
        from decision_engine.shock_simulator import ShockSimulator, ShockEvent, ShockType

        sim = ShockSimulator(self.adj_matrix, self.node_names, n_monte_carlo=300)

        events = [ShockEvent(ShockType.SECTOR_CRASH, magnitude=magnitude,
                             affected_nodes=self.node_names[:2], step=5)]
        shock_result = sim.run_scenario(events, n_steps=40)
        mc_results = sim.monte_carlo_shock(events[0], n_steps=30)
        return shock_result, mc_results

    def _run_scoring(self, contagion_result, shock_result):
        from decision_engine.investment_scoring import InvestmentScoringEngine, StockSignals
        engine = InvestmentScoringEngine()
        signals_list = []

        for i, ticker in enumerate(self.node_names):
            # Get centrality
            cent = {}
            if self.centrality_scores is not None:
                row = self.centrality_scores[self.centrality_scores["ticker"] == ticker]
                if len(row):
                    cent = row.iloc[0].to_dict()

            # Get fundamentals
            fund = self.fundamentals.get(ticker, {})

            # Get contagion stress
            contagion_stress = 0.0
            if ticker in contagion_result.stress_levels:
                contagion_stress = contagion_result.stress_levels[ticker][-1]

            s = StockSignals(
                ticker=ticker,
                momentum_1m=float(self.returns_df[ticker].tail(21).sum())
                            if self.returns_df is not None else 0.0,
                momentum_3m=float(self.returns_df[ticker].tail(63).sum())
                            if self.returns_df is not None else 0.0,
                volatility=float(self.returns_df[ticker].std() * np.sqrt(252))
                           if self.returns_df is not None else 0.02,
                rsi=float(fund.get("rsi", 50.0)),
                pe_ratio=float(fund.get("pe_ratio", 20.0)),
                pb_ratio=float(fund.get("pb_ratio", 2.0)),
                degree_centrality=float(cent.get("degree", 0.3)),
                betweenness_centrality=float(cent.get("betweenness", 0.1)),
                eigenvector_centrality=float(cent.get("eigenvector", 0.2)),
                gnn_crash_probability=float(self.crash_probabilities.get(ticker, 0.2))
                                      if self.crash_probabilities else 0.2,
                gnn_predicted_return=float(self.predicted_returns.get(ticker, 0.001))
                                     if self.predicted_returns else 0.001,
                contagion_stress=float(contagion_stress),
                shock_max_drawdown=float(shock_result.max_drawdown),
                var_95=float(shock_result.var_95),
                cvar_95=float(shock_result.cvar_95),
            )
            signals_list.append(s)

        return engine.rank_stocks(signals_list)

    def _run_rl(self, n_episodes):
        from decision_engine.portfolio_rl import PortfolioEnv, RLPortfolioOptimizer

        env = PortfolioEnv(self.returns_df, window=min(20, len(self.returns_df) // 5))
        agent = RLPortfolioOptimizer(env, hidden_dim=32, n_episodes=n_episodes)
        agent.train()
        backtest = agent.backtest()

        # Get final weights from last state
        state = env.reset()
        weights = agent.policy.get_weights(state)
        weight_dict = {t: float(w) for t, w in zip(self.node_names, weights)}
        return weight_dict, backtest

    def _equal_weight_fallback(self):
        n = len(self.node_names)
        return {t: 1.0 / n for t in self.node_names}

    def _validate(self):
        if self.adj_matrix is None or self.node_names is None:
            raise ValueError("Graph data not loaded. Call load_graph_data() first.")
        self._is_ready = True

   
    def format_portfolio_report(self, results: Dict) -> str:
    
        lines = [
            "=" * 60,
            "  DECISION ENGINE — FULL PIPELINE REPORT",
            "=" * 60,
            "",
            "── INVESTMENT RANKINGS ──",
        ]

        if "investment_rankings" in results:
            df = results["investment_rankings"]
            lines.append(df[["rank", "ticker", "composite_score", "recommendation"]]
                         .to_string(index=False))

        if "contagion_result" in results:
            cr = results["contagion_result"]
            lines += [
                "",
                "── CONTAGION SIMULATION ──",
                f"Systemic Risk Score : {cr.systemic_risk_score:.4f}",
                f"Cascade Depth       : {cr.cascade_depth} steps",
                f"Final Infected      : {cr.final_infected}",
            ]

        if "shock_result" in results:
            sr = results["shock_result"]
            lines += [
                "",
                "── SHOCK SIMULATION ──",
                f"Max Drawdown : {sr.max_drawdown:.4f}",
                f"VaR  (95%)   : {sr.var_95:.4f}",
                f"CVaR (95%)   : {sr.cvar_95:.4f}",
                f"Recovery     : {sr.recovery_time} steps",
            ]

        if "portfolio_weights" in results:
            lines += ["", "── PORTFOLIO WEIGHTS ──"]
            for t, w in sorted(results["portfolio_weights"].items(),
                                key=lambda x: -x[1]):
                lines.append(f"  {t}: {w:.4f} ({w*100:.1f}%)")

        lines += ["", "=" * 60]
        return "\n".join(lines)


def quick_demo():
    np.random.seed(42)
    tickers = ["AAPL", "MSFT", "JPM", "GS", "XOM", "CVX"]
    n = len(tickers)
    T = 100

    adj = np.random.uniform(0, 0.6, (n, n))
    adj = (adj + adj.T) / 2
    np.fill_diagonal(adj, 0)

    returns_df = pd.DataFrame(
        np.random.normal(0.0005, 0.015, (T, n)), columns=tickers
    )
    crash_probs = {t: np.random.uniform(0.05, 0.5) for t in tickers}
    pred_returns = {t: np.random.uniform(-0.02, 0.04) for t in tickers}

    di = DecisionInterface()
    di.load_graph_data(adj, tickers)
    di.load_ai_predictions(crash_probs, pred_returns)
    di.load_market_data(returns_df)

    results = di.run_full_pipeline(n_rl_episodes=10)
    print(di.format_portfolio_report(results))
    return results


if __name__ == "__main__":
    quick_demo()