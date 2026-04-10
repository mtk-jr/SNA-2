
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class PortfolioEnv:
 

    def __init__(
        self,
        returns_df: pd.DataFrame,
        risk_scores: Optional[np.ndarray] = None,
        window: int = 20,
        transaction_cost: float = 0.001,
    ):
        self.returns = returns_df.values.astype(float)  # (T, n)
        self.tickers = list(returns_df.columns)
        self.n = len(self.tickers)
        self.T = len(self.returns)
        self.window = window
        self.tc = transaction_cost
        self.risk_scores = risk_scores if risk_scores is not None else np.zeros((self.T, self.n))

        self.reset()

    def reset(self) -> np.ndarray:
    
        self.t = self.window
        self.weights = np.ones(self.n) / self.n  # start equal-weight
        self.portfolio_value = 1.0
        self.value_history = [1.0]
        return self._get_state()

    def _get_state(self) -> np.ndarray:
    
        ret_window = self.returns[self.t - self.window: self.t]   # (window, n)
        risk_now = self.risk_scores[self.t]                        # (n,)
        state = np.concatenate([
            ret_window.flatten(),  # (window × n)
            self.weights,          # (n,)
            risk_now,              # (n,)
        ])
        return state.astype(np.float32)

    def step(self, new_weights: np.ndarray) -> Tuple[np.ndarray, float, bool]:
 
        # Normalise weights to sum to 1 (softmax ensures all positive)
        new_weights = np.clip(new_weights, 0, None)
        new_weights = new_weights / (new_weights.sum() + 1e-8)

        # Transaction cost: proportional to weight change
        turnover = np.abs(new_weights - self.weights).sum()
        cost = self.tc * turnover

        # Apply weights to next day's returns
        day_returns = self.returns[self.t]
        portfolio_return = float(np.dot(new_weights, day_returns)) - cost

        # Update state
        self.weights = new_weights
        self.portfolio_value *= (1 + portfolio_return)
        self.value_history.append(self.portfolio_value)
        self.t += 1

        # Reward = Sharpe-like: return penalised by recent volatility
        recent_vals = np.array(self.value_history[-20:])
        recent_rets = np.diff(recent_vals) / recent_vals[:-1] if len(recent_vals) > 1 else np.array([0.0])
        sharpe = float(np.mean(recent_rets) / (np.std(recent_rets) + 1e-8))
        reward = portfolio_return + 0.1 * sharpe

        # CVaR penalty: if return is in bad tail, penalise harder
        if len(recent_rets) >= 5:
            var_5 = np.percentile(recent_rets, 5)
            cvar_penalty = -np.mean(recent_rets[recent_rets <= var_5]) * 0.5
            reward -= cvar_penalty

        done = self.t >= self.T - 1
        next_state = self._get_state() if not done else np.zeros_like(self._get_state())
        return next_state, reward, done



class PolicyNetwork:


    def __init__(self, state_dim: int, n_assets: int, hidden_dim: int = 128):
        self.state_dim = state_dim
        self.n_assets = n_assets

        # Xavier initialisation
        self.W1 = np.random.randn(state_dim, hidden_dim) * np.sqrt(2 / state_dim)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, hidden_dim) * np.sqrt(2 / hidden_dim)
        self.b2 = np.zeros(hidden_dim)
        self.W3 = np.random.randn(hidden_dim, n_assets) * np.sqrt(2 / hidden_dim)
        self.b3 = np.zeros(n_assets)

    def forward(self, state: np.ndarray) -> np.ndarray:
 
        h1 = np.tanh(state @ self.W1 + self.b1)
        h2 = np.tanh(h1 @ self.W2 + self.b2)
        logits = h2 @ self.W3 + self.b3
        return logits

    def get_weights(self, state: np.ndarray) -> np.ndarray:

        logits = self.forward(state)
        # Softmax: ensures weights are positive and sum to 1
        e = np.exp(logits - logits.max())
        return e / e.sum()

    def get_params(self) -> List[np.ndarray]:
        return [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]

    def set_params(self, params: List[np.ndarray]):
        self.W1, self.b1, self.W2, self.b2, self.W3, self.b3 = params



class RLPortfolioOptimizer:
   
    def __init__(
        self,
        env: PortfolioEnv,
        hidden_dim: int = 64,
        lr: float = 1e-3,
        n_episodes: int = 50,
        gamma: float = 0.99,
    ):
        self.env = env
        self.lr = lr
        self.n_episodes = n_episodes
        self.gamma = gamma

        # Figure out state dimension from a sample reset
        sample_state = env.reset()
        state_dim = len(sample_state)

        self.policy = PolicyNetwork(state_dim, env.n, hidden_dim)
        self.training_log: List[Dict] = []

    def _compute_returns(self, rewards: List[float]) -> np.ndarray:
  
        G = np.zeros(len(rewards))
        running = 0.0
        for t in reversed(range(len(rewards))):
            running = rewards[t] + self.gamma * running
            G[t] = running
        # Normalise for stable training
        G = (G - G.mean()) / (G.std() + 1e-8)
        return G

    def _policy_gradient_update(self, states, actions, returns):
  
        eps = 1e-4
        params = self.policy.get_params()

        for param in params:
            grad = np.zeros_like(param)
            it = np.nditer(param, flags=["multi_index"])
            while not it.finished:
                idx = it.multi_index
                original = param[idx]

                # Compute loss at param + eps
                param[idx] = original + eps
                loss_plus = self._episode_loss(states, actions, returns)

                # Compute loss at param - eps
                param[idx] = original - eps
                loss_minus = self._episode_loss(states, actions, returns)

                grad[idx] = (loss_plus - loss_minus) / (2 * eps)
                param[idx] = original
                it.iternext()

            param -= self.lr * grad   # gradient descent on loss = ascent on reward

    def _episode_loss(self, states, actions, returns) -> float:

        loss = 0.0
        for s, a, G in zip(states, actions, returns):
            pred_weights = self.policy.get_weights(s)
            # Log-prob approximation: -sum(G * log(weights))
            log_prob = np.sum(np.log(pred_weights + 1e-8) * a)
            loss -= G * log_prob
        return loss / len(states)

    def train(self) -> pd.DataFrame:
     
        logger.info(f"Training RL agent for {self.n_episodes} episodes...")

        for ep in range(self.n_episodes):
            state = self.env.reset()
            states, actions, rewards = [], [], []
            total_reward = 0.0

            while True:
                weights = self.policy.get_weights(state)
                next_state, reward, done = self.env.step(weights)

                states.append(state)
                actions.append(weights)
                rewards.append(reward)
                total_reward += reward

                state = next_state
                if done:
                    break

            # Compute discounted returns and update (only on small param subsets for speed)
            G = self._compute_returns(rewards)
            # Approximate update: only update output layer weights (fast)
            subset_states = states[::max(1, len(states)//20)]
            subset_actions = actions[::max(1, len(actions)//20)]
            subset_G = G[::max(1, len(G)//20)]

            # Gradient-free optimisation: perturb and keep better params
            self._es_update(subset_states, subset_actions, subset_G)

            final_value = self.env.portfolio_value
            log_entry = {
                "episode": ep + 1,
                "total_reward": total_reward,
                "final_portfolio_value": final_value,
                "n_steps": len(rewards),
            }
            self.training_log.append(log_entry)

            if (ep + 1) % 10 == 0:
                logger.info(f"Episode {ep+1}/{self.n_episodes} | "
                            f"Reward: {total_reward:.4f} | Value: {final_value:.4f}")

        return pd.DataFrame(self.training_log)

    def _es_update(self, states, actions, returns):
    
        sigma = 0.01
        n_samples = 10
        params = self.policy.get_params()
        flat_params = np.concatenate([p.ravel() for p in params])

        best_loss = self._episode_loss(states, actions, returns)
        best_flat = flat_params.copy()

        for _ in range(n_samples):
            noise = np.random.randn(len(flat_params)) * sigma
            candidate = flat_params + noise

            # Load candidate params into policy
            self._load_flat_params(candidate)
            loss = self._episode_loss(states, actions, returns)

            if loss < best_loss:
                best_loss = loss
                best_flat = candidate.copy()

        self._load_flat_params(best_flat)

    def _load_flat_params(self, flat: np.ndarray):
        params = self.policy.get_params()
        idx = 0
        for p in params:
            size = p.size
            p[:] = flat[idx: idx + size].reshape(p.shape)
            idx += size

    def get_optimal_weights(self, current_state: np.ndarray) -> Dict[str, float]:
      
        weights = self.policy.get_weights(current_state)
        return {ticker: float(w) for ticker, w in zip(self.env.tickers, weights)}

    def backtest(self) -> pd.DataFrame:
       
        state = self.env.reset()
        records = []

        while True:
            weights = self.policy.get_weights(state)
            next_state, reward, done = self.env.step(weights)
            records.append({
                "step": self.env.t,
                "portfolio_value": self.env.portfolio_value,
                "reward": reward,
                **{f"w_{ticker}": w for ticker, w in zip(self.env.tickers, weights)},
            })
            state = next_state
            if done:
                break

        return pd.DataFrame(records)


class BayesianPortfolioAllocator:
 

    def __init__(
        self,
        tickers: List[str],
        prior_mean: float = 0.0005,
        prior_std: float = 0.01,
    ):
        self.tickers = tickers
        self.n = len(tickers)

        # Bayesian conjugate prior: Normal distribution on returns
        self.mu    = np.full(self.n, prior_mean)   # prior mean
        self.sigma = np.full(self.n, prior_std)    # prior std (uncertainty)
        self.n_obs = np.zeros(self.n)              # observations seen

    def update(self, observed_returns: np.ndarray):
   
        for i in range(self.n):
            r = observed_returns[i]
            n = self.n_obs[i]

            # Bayesian update for Normal-Normal conjugate
            prior_precision = 1 / (self.sigma[i] ** 2 + 1e-8)
            likelihood_precision = n + 1

            posterior_precision = prior_precision + likelihood_precision
            posterior_mean = (prior_precision * self.mu[i] + likelihood_precision * r) / posterior_precision

            self.mu[i] = posterior_mean
            self.sigma[i] = np.sqrt(1 / posterior_precision)
            self.n_obs[i] += 1

    def get_weights(self, cvar_constraint: float = -0.05) -> Dict[str, float]:
     
        # Estimate 5th percentile of return distribution
        lower_bound = self.mu - 1.65 * self.sigma   # 95% confidence lower bound

        # Set weight to 0 for stocks that breach CVaR constraint
        raw_weights = np.where(lower_bound >= cvar_constraint, self.mu, 0.0)
        raw_weights = np.clip(raw_weights, 0, None)

        total = raw_weights.sum()
        if total <= 0:
            # Fallback: minimum variance (inverse sigma)
            inv_var = 1 / (self.sigma ** 2 + 1e-8)
            raw_weights = inv_var / inv_var.sum()
        else:
            raw_weights /= total

        return {t: float(w) for t, w in zip(self.tickers, raw_weights)}

    def uncertainty_report(self) -> pd.DataFrame:

        return pd.DataFrame({
            "ticker": self.tickers,
            "expected_return": self.mu,
            "uncertainty_std": self.sigma,
            "lower_95pct": self.mu - 1.65 * self.sigma,
            "observations": self.n_obs.astype(int),
        }).sort_values("expected_return", ascending=False)


if __name__ == "__main__":
    np.random.seed(42)
    tickers = ["AAPL", "MSFT", "JPM", "GS", "XOM", "AMZN"]
    n = len(tickers)
    T = 200

    # Fake return data 
    fake_returns = np.random.normal(0.0005, 0.015, (T, n))
    returns_df = pd.DataFrame(fake_returns, columns=tickers)

    # --- RL Optimizer ---
    env = PortfolioEnv(returns_df, window=10, transaction_cost=0.001)
    agent = RLPortfolioOptimizer(env, hidden_dim=32, lr=5e-3, n_episodes=20, gamma=0.95)
    train_log = agent.train()

    print("\n=== RL Training Log (last 5 episodes) ===")
    print(train_log.tail().to_string(index=False))

    backtest_df = agent.backtest()
    print(f"\nFinal Portfolio Value (RL): {backtest_df['portfolio_value'].iloc[-1]:.4f}")

    # --- Bayesian Allocator ---
    bpa = BayesianPortfolioAllocator(tickers)
    for day in range(50):
        bpa.update(fake_returns[day])

    weights = bpa.get_weights()
    print("\n=== Bayesian Portfolio Weights ===")
    for t, w in weights.items():
        print(f"  {t}: {w:.4f}")

    print("\n=== Uncertainty Report ===")
    print(bpa.uncertainty_report().to_string(index=False))