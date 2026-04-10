

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)
try:
    import dash
    from dash import dcc, html, Input, Output, State
    DASH_AVAILABLE = True
except ImportError:
    DASH_AVAILABLE = False
    logger.warning("Dash not installed. Use: pip install dash")



class DashboardDataProvider:
   

    def __init__(self, use_real_data: bool = False):
        self.use_real_data = use_real_data
        self.tickers = ["AAPL", "MSFT", "JPM", "GS", "XOM", "CVX", "AMZN", "GOOGL"]
        np.random.seed(42)

    def get_investment_rankings(self) -> pd.DataFrame:
        if self.use_real_data:
            from investment_scoring import InvestmentScoringEngine, StockSignals
            # Populate with real signals here
            pass

        # Demo data
        return pd.DataFrame({
            "ticker": self.tickers,
            "composite_score": np.random.uniform(30, 90, len(self.tickers)),
            "return_score":    np.random.uniform(20, 95, len(self.tickers)),
            "risk_score":      np.random.uniform(20, 95, len(self.tickers)),
            "contagion_score": np.random.uniform(15, 85, len(self.tickers)),
            "momentum_score":  np.random.uniform(25, 90, len(self.tickers)),
            "fundamental_score": np.random.uniform(30, 80, len(self.tickers)),
            "recommendation":  np.random.choice(["BUY", "HOLD", "AVOID"], len(self.tickers)),
        }).sort_values("composite_score", ascending=False).reset_index(drop=True)

    def get_contagion_data(self, seed_nodes: List[str], magnitude: float) -> Dict:
        n = len(self.tickers)
        stress = np.random.uniform(0, magnitude, n)
        for node in seed_nodes:
            if node in self.tickers:
                stress[self.tickers.index(node)] = min(1.0, magnitude * 1.2)

        return {
            "tickers": self.tickers,
            "stress": stress,
            "adjacency": np.random.uniform(0, 0.6, (n, n)) * (1 - np.eye(n)),
        }

    def get_portfolio_paths(self, n_paths: int = 5) -> pd.DataFrame:
        T = 100
        rows = []
        for i in range(n_paths):
            returns = np.random.normal(0.0005, 0.015, T)
            if i == 0:
                returns[30:35] -= 0.05  # simulate a crash
            values = np.cumprod(1 + returns)
            for t in range(T):
                rows.append({"step": t, "value": values[t], "path": f"Scenario {i+1}"})
        return pd.DataFrame(rows)

    def get_mc_distribution(self, n_sims: int = 500) -> pd.Series:
        returns = np.random.normal(-0.02, 0.15, n_sims)
        # Fat tails (add some extreme events)
        n_extreme = int(n_sims * 0.05)
        returns[:n_extreme] = np.random.uniform(-0.5, -0.2, n_extreme)
        return pd.Series(returns, name="total_return")

    def get_shock_comparison(self) -> pd.DataFrame:
        crises = ["2008 Financial Crisis", "COVID-19 Crash", "2022 Rate Hike", "Russia-Ukraine"]
        return pd.DataFrame({
            "crisis": crises,
            "max_drawdown": [-0.52, -0.34, -0.19, -0.12],
            "var_95":  [-0.031, -0.022, -0.014, -0.009],
            "cvar_95": [-0.055, -0.038, -0.024, -0.016],
            "recovery_steps": [45, 22, 38, 15],
        })


class ChartBuilder:
    COLORS = {
        "BUY":   "#00C851",
        "HOLD":  "#FFD700",
        "AVOID": "#FF4444",
        "primary": "#1E88E5",
        "bg": "#0D1117",
        "card_bg": "#161B22",
        "text": "#E6EDF3",
    }

    def investment_ranking_chart(self, df: pd.DataFrame) -> go.Figure:
        color_map = {"BUY": self.COLORS["BUY"],
                     "HOLD": self.COLORS["HOLD"],
                     "AVOID": self.COLORS["AVOID"]}
        colors = [color_map.get(r, "#888") for r in df["recommendation"]]

        fig = go.Figure(go.Bar(
            x=df["composite_score"],
            y=df["ticker"],
            orientation="h",
            marker_color=colors,
            text=[f"{s:.1f} — {r}" for s, r in zip(df["composite_score"], df["recommendation"])],
            textposition="inside",
            hovertemplate="<b>%{y}</b><br>Score: %{x:.1f}<extra></extra>",
        ))
        fig.update_layout(
            title="📊 Investment Rankings (Composite Score)",
            xaxis_title="Score (0–100)",
            yaxis=dict(autorange="reversed"),
            paper_bgcolor=self.COLORS["bg"],
            plot_bgcolor=self.COLORS["card_bg"],
            font_color=self.COLORS["text"],
            height=400,
        )
        return fig

    def score_radar_chart(self, df: pd.DataFrame, ticker: str) -> go.Figure:
        row = df[df["ticker"] == ticker].iloc[0]
        dims = ["return_score", "risk_score", "contagion_score", "momentum_score", "fundamental_score"]
        labels = ["Return", "Risk Safety", "Contagion Safety", "Momentum", "Fundamentals"]
        values = [row[d] for d in dims]
        values.append(values[0])  # close the loop

        fig = go.Figure(go.Scatterpolar(
            r=values,
            theta=labels + [labels[0]],
            fill="toself",
            fillcolor="rgba(30, 136, 229, 0.3)",
            line_color=self.COLORS["primary"],
            name=ticker,
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(range=[0, 100], visible=True)),
            title=f"🕷️ Score Breakdown — {ticker}",
            paper_bgcolor=self.COLORS["bg"],
            font_color=self.COLORS["text"],
            height=350,
        )
        return fig

    def contagion_network_chart(self, contagion_data: Dict) -> go.Figure:
        tickers = contagion_data["tickers"]
        stress = contagion_data["stress"]
        adj = contagion_data["adjacency"]
        n = len(tickers)

        # Simple circular layout
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        x_pos = np.cos(angles)
        y_pos = np.sin(angles)

        # Build edge traces
        edge_traces = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i, j] > 0.3:  # only show strong connections
                    edge_traces.append(go.Scatter(
                        x=[x_pos[i], x_pos[j], None],
                        y=[y_pos[i], y_pos[j], None],
                        mode="lines",
                        line=dict(width=adj[i, j] * 2, color="rgba(100,100,200,0.3)"),
                        hoverinfo="none",
                        showlegend=False,
                    ))

        # Node trace
        node_trace = go.Scatter(
            x=x_pos, y=y_pos,
            mode="markers+text",
            text=tickers,
            textposition="top center",
            marker=dict(
                size=20,
                color=stress,
                colorscale="RdYlGn_r",
                cmin=0, cmax=1,
                colorbar=dict(title="Stress", thickness=10),
                line=dict(width=2, color="white"),
            ),
            hovertemplate="<b>%{text}</b><br>Stress: %{marker.color:.3f}<extra></extra>",
        )

        fig = go.Figure(data=edge_traces + [node_trace])
        fig.update_layout(
            title="🌐 Contagion Network (Red = High Stress)",
            showlegend=False,
            paper_bgcolor=self.COLORS["bg"],
            plot_bgcolor=self.COLORS["card_bg"],
            font_color=self.COLORS["text"],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=450,
        )
        return fig

    def portfolio_path_chart(self, paths_df: pd.DataFrame) -> go.Figure:
        fig = px.line(
            paths_df, x="step", y="value", color="path",
            title="📈 Portfolio Value Paths (RL Agent Simulation)",
            labels={"step": "Trading Day", "value": "Portfolio Value (normalised)"},
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig.add_hline(y=1.0, line_dash="dash", line_color="gray",
                      annotation_text="Initial Value")
        fig.update_layout(
            paper_bgcolor=self.COLORS["bg"],
            plot_bgcolor=self.COLORS["card_bg"],
            font_color=self.COLORS["text"],
            height=350,
        )
        return fig

    def mc_distribution_chart(self, returns: pd.Series) -> go.Figure:
        var_95  = float(np.percentile(returns, 5))
        cvar_95 = float(returns[returns <= var_95].mean())

        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=returns, nbinsx=50,
            marker_color=self.COLORS["primary"],
            opacity=0.7, name="Return Distribution"
        ))
        fig.add_vline(x=var_95, line_color="orange", line_dash="dash",
                      annotation_text=f"VaR(95%): {var_95:.3f}")
        fig.add_vline(x=cvar_95, line_color="red", line_dash="dash",
                      annotation_text=f"CVaR(95%): {cvar_95:.3f}")
        fig.update_layout(
            title="🎲 Monte Carlo Return Distribution",
            xaxis_title="Total Return",
            yaxis_title="Frequency",
            paper_bgcolor=self.COLORS["bg"],
            plot_bgcolor=self.COLORS["card_bg"],
            font_color=self.COLORS["text"],
            height=350,
        )
        return fig

    def shock_heatmap(self, shock_df: pd.DataFrame) -> go.Figure:
        """Heatmap comparing historical crisis impacts."""
        metrics = ["max_drawdown", "var_95", "cvar_95"]
        labels  = ["Max Drawdown", "VaR (95%)", "CVaR (95%)"]
        z = shock_df[metrics].values

        fig = go.Figure(go.Heatmap(
            z=z.T,
            x=shock_df["crisis"],
            y=labels,
            colorscale="RdYlGn_r",
            text=np.round(z.T, 4),
            texttemplate="%{text}",
            hovertemplate="<b>%{y}</b><br>%{x}<br>Value: %{z:.4f}<extra></extra>",
        ))
        fig.update_layout(
            title="🔥 Historical Crisis Stress Test Comparison",
            paper_bgcolor=self.COLORS["bg"],
            plot_bgcolor=self.COLORS["card_bg"],
            font_color=self.COLORS["text"],
            height=300,
        )
        return fig



def create_dash_app(data_provider: Optional[DashboardDataProvider] = None) -> "dash.Dash":
 
    if not DASH_AVAILABLE:
        raise ImportError("Install Dash: pip install dash")

    dp = data_provider or DashboardDataProvider()
    cb = ChartBuilder()

    app = dash.Dash(__name__, title="Financial Contagion Dashboard")
    app.config.suppress_callback_exceptions = True

    app.layout = html.Div(
        style={"backgroundColor": "#0D1117", "minHeight": "100vh",
               "fontFamily": "Inter, sans-serif", "color": "#E6EDF3", "padding": "20px"},
        children=[

            # Header
            html.H1("Financial Contagion & Portfolio Intelligence Dashboard",
                    style={"textAlign": "center", "color": "#58A6FF", "marginBottom": "10px"}),
            html.P("Decision & Simulation Engine",
                   style={"textAlign": "center", "color": "#8B949E", "marginBottom": "30px"}),

            # ---- Row 1: Rankings + Radar ----
            html.Div(style={"display": "flex", "gap": "20px", "marginBottom": "20px"}, children=[
                html.Div(style={"flex": 2, "backgroundColor": "#161B22",
                                "borderRadius": "12px", "padding": "15px"}, children=[
                    dcc.Graph(id="ranking-chart"),
                ]),
                html.Div(style={"flex": 1, "backgroundColor": "#161B22",
                                "borderRadius": "12px", "padding": "15px"}, children=[
                    html.Label("Select Stock for Radar:", style={"color": "#8B949E"}),
                    dcc.Dropdown(id="radar-ticker-select",
                                 options=[{"label": t, "value": t} for t in dp.tickers],
                                 value=dp.tickers[0],
                                 style={"backgroundColor": "#0D1117", "color": "#E6EDF3"}),
                    dcc.Graph(id="radar-chart"),
                ]),
            ]),

            # ---- Row 2: Contagion Network ----
            html.Div(style={"backgroundColor": "#161B22", "borderRadius": "12px",
                            "padding": "15px", "marginBottom": "20px"}, children=[
                html.Div(style={"display": "flex", "gap": "20px", "alignItems": "center",
                                "marginBottom": "15px"}, children=[
                    html.Div(style={"flex": 2}, children=[
                        html.Label("Seed Nodes (shock origin):", style={"color": "#8B949E"}),
                        dcc.Dropdown(id="seed-nodes-select",
                                     options=[{"label": t, "value": t} for t in dp.tickers],
                                     value=["JPM", "GS"], multi=True,
                                     style={"backgroundColor": "#0D1117"}),
                    ]),
                    html.Div(style={"flex": 1}, children=[
                        html.Label("Shock Magnitude:", style={"color": "#8B949E"}),
                        dcc.Slider(id="shock-magnitude", min=0.1, max=1.0, step=0.1,
                                   value=0.5, marks={i/10: str(i/10) for i in range(1, 11)}),
                    ]),
                ]),
                dcc.Graph(id="contagion-network"),
            ]),

            # ---- Row 3: Portfolio Paths + MC Distribution ----
            html.Div(style={"display": "flex", "gap": "20px", "marginBottom": "20px"}, children=[
                html.Div(style={"flex": 1, "backgroundColor": "#161B22",
                                "borderRadius": "12px", "padding": "15px"}, children=[
                    dcc.Graph(id="portfolio-paths"),
                ]),
                html.Div(style={"flex": 1, "backgroundColor": "#161B22",
                                "borderRadius": "12px", "padding": "15px"}, children=[
                    dcc.Graph(id="mc-distribution"),
                ]),
            ]),

            # ---- Row 4: Historical Stress Test ----
            html.Div(style={"backgroundColor": "#161B22", "borderRadius": "12px",
                            "padding": "15px", "marginBottom": "20px"}, children=[
                dcc.Graph(id="shock-heatmap"),
            ]),

            # Hidden stores
            dcc.Store(id="rankings-store"),
        ]
    )



    @app.callback(
        Output("ranking-chart", "figure"),
        Output("rankings-store", "data"),
        Input("ranking-chart", "id"),   # fires on load
    )
    def load_rankings(_):
        df = dp.get_investment_rankings()
        return cb.investment_ranking_chart(df), df.to_json()

    @app.callback(
        Output("radar-chart", "figure"),
        Input("radar-ticker-select", "value"),
        State("rankings-store", "data"),
    )
    def update_radar(ticker, rankings_json):
        if not rankings_json:
            return go.Figure()
        df = pd.read_json(rankings_json)
        return cb.score_radar_chart(df, ticker)

    @app.callback(
        Output("contagion-network", "figure"),
        Input("seed-nodes-select", "value"),
        Input("shock-magnitude", "value"),
    )
    def update_contagion(seed_nodes, magnitude):
        data = dp.get_contagion_data(seed_nodes or [], magnitude or 0.5)
        return cb.contagion_network_chart(data)

    @app.callback(
        Output("portfolio-paths", "figure"),
        Input("portfolio-paths", "id"),
    )
    def load_portfolio_paths(_):
        paths = dp.get_portfolio_paths(n_paths=5)
        return cb.portfolio_path_chart(paths)

    @app.callback(
        Output("mc-distribution", "figure"),
        Input("mc-distribution", "id"),
    )
    def load_mc_distribution(_):
        returns = dp.get_mc_distribution(n_sims=500)
        return cb.mc_distribution_chart(returns)

    @app.callback(
        Output("shock-heatmap", "figure"),
        Input("shock-heatmap", "id"),
    )
    def load_shock_heatmap(_):
        shock_df = dp.get_shock_comparison()
        return cb.shock_heatmap(shock_df)

    logger.info("Dash app created successfully")
    return app


def export_static_charts(output_dir: str = "charts"):
  
    import os
    os.makedirs(output_dir, exist_ok=True)

    dp = DashboardDataProvider()
    cb = ChartBuilder()

    charts = {
        "rankings": cb.investment_ranking_chart(dp.get_investment_rankings()),
        "contagion": cb.contagion_network_chart(dp.get_contagion_data(["JPM"], 0.6)),
        "portfolio": cb.portfolio_path_chart(dp.get_portfolio_paths()),
        "mc_dist":   cb.mc_distribution_chart(dp.get_mc_distribution()),
        "shock":     cb.shock_heatmap(dp.get_shock_comparison()),
    }

    for name, fig in charts.items():
        path = os.path.join(output_dir, f"{name}.html")
        fig.write_html(path)
        print(f"Saved: {path}")

    print(f"\nAll charts saved to ./{output_dir}/")



if __name__ == "__main__":
    if DASH_AVAILABLE:
        app = create_dash_app()
        print("\n Dashboard starting at http://localhost:8050")
        print("   Press Ctrl+C to stop.\n")
        app.run(debug=True, host="0.0.0.0", port=8050)
    else:
        print("Dash not installed. Exporting static charts instead...")
        export_static_charts()