

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)



@dataclass
class StockSignals:
   
    ticker: str
    momentum_1m: float = 0.0       # 1-month return
    momentum_3m: float = 0.0       # 3-month return
    volatility: float = 0.01       # annualised volatility
    rsi: float = 50.0              # RSI indicator (0-100)
    pe_ratio: float = 20.0         # Price-to-Earnings
    pb_ratio: float = 2.0          # Price-to-Book

    degree_centrality: float = 0.0         # how many connections this stock has
    betweenness_centrality: float = 0.0    # bridge node? higher = more systemic
    eigenvector_centrality: float = 0.0    # connected to important nodes?


    gnn_crash_probability: float = 0.0    # probability of crash in next period
    gnn_predicted_return: float = 0.0     # predicted future return

    contagion_stress: float = 0.0         # final stress level from contagion sim
    shock_max_drawdown: float = 0.0       # worst drawdown under shock scenarios
    var_95: float = 0.0                   # Value at Risk
    cvar_95: float = 0.0                  # Conditional VaR (tail risk)


@dataclass
class InvestmentScore:
    ticker: str
    return_score: float        # 0-100
    risk_score: float          # 0-100 (100 = safest)
    contagion_score: float     # 0-100 (100 = least exposed)
    momentum_score: float      # 0-100
    fundamental_score: float   # 0-100
    composite_score: float     # final weighted score 0-100
    rank: int = 0
    recommendation: str = ""   # BUY / HOLD / AVOID
    score_breakdown: Dict[str, float] = field(default_factory=dict)



class InvestmentScoringEngine:
 

    DEFAULT_WEIGHTS = {
        "return_score":      0.25,
        "risk_score":        0.25,
        "contagion_score":   0.20,
        "momentum_score":    0.15,
        "fundamental_score": 0.15,
    }

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        cvar_penalty_multiplier: float = 1.5,
    ):
        self.weights = weights or self.DEFAULT_WEIGHTS
        assert abs(sum(self.weights.values()) - 1.0) < 1e-6, "Weights must sum to 1"
        self.cvar_penalty = cvar_penalty_multiplier
        logger.info("InvestmentScoringEngine initialised")

    def _score_return(self, signals: StockSignals) -> float:
        expected = 0.6 * signals.gnn_predicted_return + 0.4 * signals.momentum_1m

        raw = (expected + 0.10) / 0.20 * 100
        return float(np.clip(raw, 0, 100))

    def _score_risk(self, signals: StockSignals) -> float:
 
        crash_risk = signals.gnn_crash_probability          # 0-1, higher = riskier
        vol_risk   = min(signals.volatility / 0.5, 1.0)    # normalise to 0-1
        cvar_risk  = min(abs(signals.cvar_95) / 0.20, 1.0) * self.cvar_penalty
        cvar_risk  = min(cvar_risk, 1.0)

        composite_risk = 0.4 * crash_risk + 0.3 * vol_risk + 0.3 * cvar_risk
        return float(np.clip((1 - composite_risk) * 100, 0, 100))

    def _score_contagion(self, signals: StockSignals) -> float:
    
        centrality_risk = (
            0.4 * signals.betweenness_centrality +
            0.3 * signals.eigenvector_centrality +
            0.3 * signals.degree_centrality
        )
        centrality_risk = min(centrality_risk, 1.0)
        sim_risk = min(signals.contagion_stress, 1.0)
        dd_risk  = min(abs(signals.shock_max_drawdown), 1.0)

        total_risk = 0.4 * centrality_risk + 0.35 * sim_risk + 0.25 * dd_risk
        return float(np.clip((1 - total_risk) * 100, 0, 100))

    def _score_momentum(self, signals: StockSignals) -> float:
       
        # Momentum component
        mom_raw = 0.6 * signals.momentum_1m + 0.4 * signals.momentum_3m
        mom_score = np.clip((mom_raw + 0.15) / 0.30 * 100, 0, 100)

        # RSI component: penalise overbought (>70) and reward oversold (<30)
        if signals.rsi < 30:
            rsi_score = 80   # oversold = potential buy
        elif signals.rsi > 70:
            rsi_score = 30   # overbought = cautious
        else:
            rsi_score = 50 + (50 - signals.rsi)  # neutral range

        return float(np.clip(0.7 * mom_score + 0.3 * rsi_score, 0, 100))

    def _score_fundamentals(self, signals: StockSignals) -> float:
     
        pe_score = np.clip((50 - signals.pe_ratio) / 40 * 100, 0, 100)
        pb_score = np.clip((7 - signals.pb_ratio) / 6 * 100, 0, 100)

        return float(np.clip(0.5 * pe_score + 0.5 * pb_score, 0, 100))


    def score_stock(self, signals: StockSignals) -> InvestmentScore:
        """Compute full investment score for one stock."""
        r_score  = self._score_return(signals)
        ri_score = self._score_risk(signals)
        c_score  = self._score_contagion(signals)
        m_score  = self._score_momentum(signals)
        f_score  = self._score_fundamentals(signals)

        composite = (
            self.weights["return_score"]      * r_score +
            self.weights["risk_score"]        * ri_score +
            self.weights["contagion_score"]   * c_score +
            self.weights["momentum_score"]    * m_score +
            self.weights["fundamental_score"] * f_score
        )

        recommendation = (
            "BUY"   if composite >= 65 else
            "HOLD"  if composite >= 40 else
            "AVOID"
        )

        return InvestmentScore(
            ticker=signals.ticker,
            return_score=r_score,
            risk_score=ri_score,
            contagion_score=c_score,
            momentum_score=m_score,
            fundamental_score=f_score,
            composite_score=float(composite),
            recommendation=recommendation,
            score_breakdown={
                "return":      r_score,
                "risk":        ri_score,
                "contagion":   c_score,
                "momentum":    m_score,
                "fundamental": f_score,
            },
        )

    def rank_stocks(self, signals_list: List[StockSignals]) -> pd.DataFrame:

        scores = [self.score_stock(s) for s in signals_list]
        scores.sort(key=lambda x: x.composite_score, reverse=True)

        for rank, score in enumerate(scores, 1):
            score.rank = rank

        rows = []
        for s in scores:
            rows.append({
                "rank": s.rank,
                "ticker": s.ticker,
                "composite_score": round(s.composite_score, 2),
                "return_score": round(s.return_score, 2),
                "risk_score": round(s.risk_score, 2),
                "contagion_score": round(s.contagion_score, 2),
                "momentum_score": round(s.momentum_score, 2),
                "fundamental_score": round(s.fundamental_score, 2),
                "recommendation": s.recommendation,
            })

        return pd.DataFrame(rows)

    def cvar_adjusted_portfolio(
        self,
        signals_list: List[StockSignals],
        max_cvar: float = -0.03,
        min_score: float = 40.0,
    ) -> Dict[str, float]:
        
    
        eligible = []
        for s in signals_list:
            score = self.score_stock(s)
            passes_score = score.composite_score >= min_score
            passes_cvar  = s.cvar_95 >= max_cvar
            if passes_score and passes_cvar:
                eligible.append((s.ticker, score.composite_score))

        if not eligible:
            logger.warning("No stocks passed CVaR filter. Relaxing constraints.")
            eligible = [(s.ticker, self.score_stock(s).composite_score)
                        for s in signals_list]

        total_score = sum(sc for _, sc in eligible)
        weights = {t: sc / total_score for t, sc in eligible}
        logger.info(f"CVaR-adjusted portfolio: {len(weights)} stocks selected")
        return weights

    def agent_based_price_impact(
        self,
        signals_list: List[StockSignals],
        n_agents: int = 200,
        n_steps: int = 50,
    ) -> pd.DataFrame:
      
        np.random.seed(99)
        n_stocks = len(signals_list)
        prices = np.ones(n_stocks)  # normalised starting price = 1
        fundamentals = np.array([max(0.5, 1 + s.gnn_predicted_return * 10)
                                  for s in signals_list])
        momentums = np.array([s.momentum_1m for s in signals_list])

        n_fund = int(n_agents * 0.4)
        n_mom  = int(n_agents * 0.3)
        n_noise= n_agents - n_fund - n_mom

        price_paths = [prices.copy()]

        for step in range(n_steps):
            demands = np.zeros(n_stocks)

            # Fundamental agents: buy if cheap, sell if expensive
            for _ in range(n_fund):
                diff = fundamentals - prices
                demands += diff * np.random.uniform(0.5, 1.5, n_stocks)

            # Momentum agents: follow recent price trend
            if len(price_paths) >= 2:
                trend = price_paths[-1] - price_paths[-2]
            else:
                trend = momentums

            for _ in range(n_mom):
                demands += trend * np.random.uniform(0.3, 1.2, n_stocks)

            # Noise traders: random
            demands += np.random.normal(0, 0.1, n_stocks) * n_noise

            # Price update: demand pushes price up/down
            price_change = demands / (n_agents * 10)
            prices = np.clip(prices + price_change, 0.01, None)
            price_paths.append(prices.copy())

        price_array = np.array(price_paths)
        tickers = [s.ticker for s in signals_list]
        result = pd.DataFrame(price_array, columns=tickers)
        result["step"] = range(len(result))

        logger.info(f"Agent-based simulation: {n_agents} agents, {n_steps} steps")
        return result



if __name__ == "__main__":
    np.random.seed(7)

    # Simulate receiving signals from all other modules
    tickers = ["AAPL", "MSFT", "JPM", "GS", "XOM", "CVX", "AMZN", "GOOGL"]
    signals = []
    for ticker in tickers:
        s = StockSignals(
            ticker=ticker,
            momentum_1m=np.random.uniform(-0.05, 0.10),
            momentum_3m=np.random.uniform(-0.10, 0.15),
            volatility=np.random.uniform(0.10, 0.40),
            rsi=np.random.uniform(25, 75),
            pe_ratio=np.random.uniform(10, 45),
            pb_ratio=np.random.uniform(0.8, 6.0),
            degree_centrality=np.random.uniform(0.2, 0.8),
            betweenness_centrality=np.random.uniform(0.0, 0.4),
            eigenvector_centrality=np.random.uniform(0.1, 0.7),
            gnn_crash_probability=np.random.uniform(0.05, 0.60),
            gnn_predicted_return=np.random.uniform(-0.03, 0.06),
            contagion_stress=np.random.uniform(0.0, 0.7),
            shock_max_drawdown=np.random.uniform(-0.25, -0.02),
            var_95=np.random.uniform(-0.05, -0.01),
            cvar_95=np.random.uniform(-0.08, -0.02),
        )
        signals.append(s)

    engine = InvestmentScoringEngine()

    print("=== Investment Rankings ===")
    rankings = engine.rank_stocks(signals)
    print(rankings.to_string(index=False))

    print("\n=== CVaR-Adjusted Portfolio ===")
    portfolio = engine.cvar_adjusted_portfolio(signals, max_cvar=-0.05, min_score=35)
    for t, w in portfolio.items():
        print(f"  {t}: {w:.4f}")

    print("\n=== Agent-Based Price Simulation (first 5 rows) ===")
    ab = engine.agent_based_price_impact(signals, n_agents=100, n_steps=20)
    print(ab.head().to_string(index=False))