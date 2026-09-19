
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.covariance import LedoitWolf
from scipy.optimize import minimize

# ============================================================
# COMEX / PPG — BLACK-LITTERMAN ASSET OPTIMIZATION
# BASIC ACADEMIC VERSION
#
# IMPORTANT METHODOLOGICAL SCOPE
# ------------------------------
# COMEX is not a separately listed PPG reportable segment.
# PPG identifies COMEX as a primary brand within its
# "Global Architectural Coatings" segment.
#
# Therefore, this academic model uses the three PPG reportable
# operating segments as asset sleeves:
#   1) Global Architectural Coatings  <-- COMEX exposure
#   2) Performance Coatings
#   3) Industrial Coatings
#
# "Returns" are NOT stock-market returns. They are a financial
# return proxy defined as:
#
#       Segment income / segment assets
#
# This makes Black-Litterman usable with public financial
# statements at a BASIC sophistication level, while clearly
# labeling the proxy.
#
# Sources: PPG 2025 Annual Report / Form 10-K and 2024 Annual
# Report / Form 10-K.
# ============================================================

st.set_page_config(
    page_title="COMEX — Black-Litterman",
    page_icon="📊",
    layout="wide",
)

# -----------------------------
# Public financial-statement data
# USD millions
# -----------------------------
YEARS = [2022, 2023, 2024, 2025]

DATA = pd.DataFrame(
    {
        "Global Architectural Coatings": {
            2022: {"sales": 3852, "income": 558, "assets": 6149, "capex": 86},
            2023: {"sales": 4021, "income": 673, "assets": 6595, "capex": 93},
            2024: {"sales": 3921, "income": 678, "assets": 5887, "capex": 160},
            2025: {"sales": 3838, "income": 599, "assets": 6676, "capex": 126},
        },
        "Performance Coatings": {
            2022: {"sales": 4792, "income": 847, "assets": 5612, "capex": 136},
            2023: {"sales": 5132, "income": 1019, "assets": 5586, "capex": 217},
            2024: {"sales": 5237, "income": 1142, "assets": 5601, "capex": 166},
            2025: {"sales": 5513, "income": 1148, "assets": 5535, "capex": 249},
        },
        "Industrial Coatings": {
            2022: {"sales": 6970, "income": 646, "assets": 5802, "capex": 313},
            2023: {"sales": 7089, "income": 968, "assets": 5643, "capex": 184},
            2024: {"sales": 6687, "income": 893, "assets": 5230, "capex": 247},
            2025: {"sales": 6524, "income": 875, "assets": 6183, "capex": 245},
        },
    }
)

ASSETS = list(DATA.columns)

# -----------------------------
# Financial return proxy
# -----------------------------
def build_roa():
    rows = []
    for year in YEARS:
        row = {}
        for asset in ASSETS:
            x = DATA.loc[year, asset]
            row[asset] = x["income"] / x["assets"]
        rows.append(pd.Series(row, name=year))
    return pd.DataFrame(rows)

ROA = build_roa()

# -----------------------------
# Helper functions
# -----------------------------
def sample_cov(returns):
    return returns.cov().values

def lw_cov(returns):
    return LedoitWolf().fit(returns.values).covariance_

def bootstrap_tau(returns, n_boot=500, seed=42):
    rng = np.random.default_rng(seed)
    values = returns.values
    n = len(values)
    boot_means = []

    for _ in range(n_boot):
        sample = values[rng.integers(0, n, size=n)]
        boot_means.append(sample.mean(axis=0))

    boot_means = np.asarray(boot_means)
    # Scale uncertainty relative to variance of the bootstrap mean.
    # Kept bounded for a stable academic model.
    tau = np.mean(np.var(boot_means, axis=0)) / max(
        np.mean(np.diag(np.cov(values, rowvar=False))), 1e-12
    )

    return float(np.clip(tau, 0.01, 1.0))

def implied_equilibrium_returns(cov, weights, risk_aversion):
    # Pi = delta * Sigma * w
    return risk_aversion * cov @ weights

def he_litterman_omega(tau, P, cov):
    # Omega = diag(P * tau Sigma * P')
    middle = tau * P @ cov @ P.T
    return np.diag(np.diag(middle))

def black_litterman_posterior(
    prior_returns,
    cov,
    tau,
    P,
    Q,
    omega,
):
    tau_cov_inv = np.linalg.inv(tau * cov)
    omega_inv = np.linalg.inv(omega)

    posterior_cov_inv = tau_cov_inv + P.T @ omega_inv @ P
    posterior_cov = np.linalg.inv(posterior_cov_inv)

    posterior_returns = posterior_cov @ (
        tau_cov_inv @ prior_returns
        + P.T @ omega_inv @ Q
    )

    return posterior_returns, posterior_cov

def optimize_max_sharpe(mu, cov, rf, min_w, max_w):
    n = len(mu)

    def neg_sharpe(w):
        port_ret = w @ mu
        port_vol = np.sqrt(max(w @ cov @ w, 1e-12))
        return -(port_ret - rf) / port_vol

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(min_w, max_w)] * n
    x0 = np.repeat(1 / n, n)

    res = minimize(
        neg_sharpe,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 2000, "ftol": 1e-12},
    )

    if not res.success:
        return x0

    return res.x

def optimize_min_vol(mu, cov, min_w, max_w):
    n = len(mu)

    def vol(w):
        return np.sqrt(max(w @ cov @ w, 1e-12))

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(min_w, max_w)] * n
    x0 = np.repeat(1 / n, n)

    res = minimize(
        vol,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 2000, "ftol": 1e-12},
    )

    if not res.success:
        return x0

    return res.x

def optimize_max_utility(mu, cov, delta, min_w, max_w):
    n = len(mu)

    def neg_utility(w):
        return -(w @ mu - 0.5 * delta * (w @ cov @ w))

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(min_w, max_w)] * n
    x0 = np.repeat(1 / n, n)

    res = minimize(
        neg_utility,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 2000, "ftol": 1e-12},
    )

    if not res.success:
        return x0

    return res.x

def portfolio_stats(w, mu, cov, rf):
    ret = float(w @ mu)
    vol = float(np.sqrt(max(w @ cov @ w, 1e-12)))
    sharpe = (ret - rf) / vol if vol > 0 else np.nan
    return ret, vol, sharpe

def efficient_frontier(mu, cov, min_w, max_w, points=60):
    n = len(mu)
    target_returns = np.linspace(mu.min(), mu.max(), points)
    vols = []
    rets = []
    weights = []

    for target in target_returns:
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},
            {"type": "eq", "fun": lambda w, t=target: w @ mu - t},
        ]
        bounds = [(min_w, max_w)] * n
        x0 = np.repeat(1 / n, n)

        res = minimize(
            lambda w: w @ cov @ w,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-12},
        )

        if res.success:
            vols.append(np.sqrt(max(res.fun, 0)))
            rets.append(target)
            weights.append(res.x)

    return np.array(rets), np.array(vols), weights

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("Black-Litterman settings")

    window = st.selectbox(
        "Historical window",
        ["4 years (2022–2025)", "3 years (2023–2025)", "2 years (2024–2025)"],
        index=0,
    )

    market_portfolio = st.selectbox(
        "Market portfolio",
        ["Asset-value weighted", "Equal Weight"],
        index=0,
    )

    risk_free = st.number_input(
        "Risk-free rate",
        min_value=-0.05,
        max_value=0.15,
        value=0.04,
        step=0.005,
        format="%.2f",
    )

    risk_aversion_mode = st.selectbox(
        "Risk aversion",
        ["Market implied", "Fixed"],
        index=0,
    )

    fixed_delta = st.number_input(
        "Fixed risk aversion",
        min_value=0.1,
        max_value=10.0,
        value=2.5,
        step=0.1,
    )

    covariance_method = st.selectbox(
        "Covariance",
        ["Ledoit-Wolf", "Sample"],
        index=0,
    )

    tau_mode = st.selectbox(
        "Tau",
        ["Statistical / bootstrap", "Fixed"],
        index=0,
    )

    fixed_tau = st.number_input(
        "Fixed tau",
        min_value=0.001,
        max_value=1.0,
        value=0.05,
        step=0.01,
    )

    omega_method = st.selectbox(
        "Omega",
        ["He-Litterman", "Identity"],
        index=0,
    )

    optimization = st.selectbox(
        "Optimization",
        ["Maximum Sharpe", "Minimum Volatility", "Maximum Utility"],
        index=0,
    )

    short_selling = st.checkbox("Allow short selling", value=False)

    if short_selling:
        min_weight = st.number_input(
            "Minimum weight",
            min_value=-1.0,
            max_value=0.0,
            value=-0.30,
            step=0.05,
        )
    else:
        min_weight = 0.0

    max_weight = st.number_input(
        "Maximum weight",
        min_value=0.05,
        max_value=1.0,
        value=0.30,
        step=0.05,
    )

# -----------------------------
# Investor views — Python only
# -----------------------------
# These are MODEL ASSUMPTIONS, not reported PPG forecasts.
# They can be changed directly in Python without a CSV.
#
# View 1: Global Architectural Coatings / COMEX exposure
# has an expected financial return proxy of 10%.
#
# View 2: Global Architectural Coatings is expected to exceed
# Industrial Coatings by 1 percentage point.
VIEWS = [
    {
        "type": "absolute",
        "asset": "Global Architectural Coatings",
        "view": 0.10,
    },
    {
        "type": "relative",
        "asset_1": "Global Architectural Coatings",
        "asset_2": "Industrial Coatings",
        "view": 0.01,
    },
]

# -----------------------------
# Select window
# -----------------------------
if window.startswith("4"):
    selected_years = [2022, 2023, 2024, 2025]
elif window.startswith("3"):
    selected_years = [2023, 2024, 2025]
else:
    selected_years = [2024, 2025]

R = ROA.loc[selected_years].copy()

# -----------------------------
# Main title
# -----------------------------
st.title("COMEX / PPG — Black-Litterman Asset Optimization")
st.caption(
    "Basic academic model using public PPG financial statements. "
    "COMEX exposure is represented by Global Architectural Coatings."
)

st.warning(
    "IMPORTANT: PPG does not disclose a complete standalone Black-Litterman "
    "portfolio for Comex. COMEX is a primary brand within Global Architectural "
    "Coatings. Therefore, the model treats PPG's three reportable operating "
    "segments as asset sleeves and uses Segment Income / Segment Assets as a "
    "financial-return proxy. These are not stock-market returns."
)

# -----------------------------
# Source data
# -----------------------------
st.subheader("1. Financial-statement data")

financial_rows = []

for year in selected_years:
    for asset in ASSETS:
        x = DATA.loc[year, asset]
        financial_rows.append(
            {
                "Year": year,
                "Asset sleeve": asset,
                "Sales (USD M)": x["sales"],
                "Segment income (USD M)": x["income"],
                "Segment assets (USD M)": x["assets"],
                "Capex (USD M)": x["capex"],
                "Return proxy": x["income"] / x["assets"],
            }
        )

financial_df = pd.DataFrame(financial_rows)

st.dataframe(
    financial_df.style.format(
        {
            "Sales (USD M)": "${:,.0f}",
            "Segment income (USD M)": "${:,.0f}",
            "Segment assets (USD M)": "${:,.0f}",
            "Capex (USD M)": "${:,.0f}",
            "Return proxy": "{:.2%}",
        }
    ),
    hide_index=True,
    use_container_width=True,
)

# -----------------------------
# Historical return proxy
# -----------------------------
st.subheader("2. Historical financial return proxy")

st.write(
    "**Return proxy = Segment income / Segment assets.** "
    "This is the basic bridge that allows the financial-statement data "
    "to enter a Black-Litterman framework."
)

st.dataframe(
    R.style.format("{:.2%}"),
    use_container_width=True,
)

# -----------------------------
# Market weights
# -----------------------------
latest_assets = pd.Series(
    {asset: DATA.loc[selected_years[-1], asset]["assets"] for asset in ASSETS}
)

if market_portfolio == "Asset-value weighted":
    market_weights = latest_assets / latest_assets.sum()
else:
    market_weights = pd.Series(1 / len(ASSETS), index=ASSETS)

st.subheader("3. Market / reference portfolio")

market_df = pd.DataFrame(
    {
        "Asset sleeve": ASSETS,
        "Reference weight": [market_weights[a] for a in ASSETS],
    }
)

st.dataframe(
    market_df.style.format({"Reference weight": "{:.2%}"}),
    hide_index=True,
    use_container_width=True,
)

# -----------------------------
# Covariance
# -----------------------------
if covariance_method == "Ledoit-Wolf":
    Sigma = lw_cov(R)
else:
    Sigma = sample_cov(R)

Sigma_df = pd.DataFrame(Sigma, index=ASSETS, columns=ASSETS)

st.subheader("4. Covariance matrix")

st.dataframe(
    Sigma_df.style.format("{:.6f}"),
    use_container_width=True,
)

# -----------------------------
# Risk aversion
# -----------------------------
if risk_aversion_mode == "Market implied":
    # Use the latest cross-sectional market portfolio and historical
    # average return proxy to derive delta.
    mu_hist = R.mean().values
    market_return = float(market_weights.values @ mu_hist)
    market_var = float(market_weights.values @ Sigma @ market_weights.values)
    delta = max((market_return - risk_free) / max(market_var, 1e-10), 0.10)
    delta = min(delta, 10.0)
else:
    delta = fixed_delta

# -----------------------------
# Tau
# -----------------------------
if tau_mode == "Statistical / bootstrap":
    tau = bootstrap_tau(R)
else:
    tau = fixed_tau

# -----------------------------
# Prior / equilibrium returns
# -----------------------------
Pi = implied_equilibrium_returns(
    Sigma,
    market_weights.values,
    delta,
)

prior_df = pd.DataFrame(
    {
        "Asset sleeve": ASSETS,
        "Equilibrium return (Pi)": Pi,
        "Historical mean proxy": R.mean().values,
    }
)

st.subheader("5. Equilibrium returns")

st.dataframe(
    prior_df.style.format(
        {
            "Equilibrium return (Pi)": "{:.2%}",
            "Historical mean proxy": "{:.2%}",
        }
    ),
    hide_index=True,
    use_container_width=True,
)

# -----------------------------
# Views matrix
# -----------------------------
P = []
Q = []

for v in VIEWS:
    row = np.zeros(len(ASSETS))

    if v["type"] == "absolute":
        row[ASSETS.index(v["asset"])] = 1.0
        Q.append(v["view"])
    else:
        row[ASSETS.index(v["asset_1"])] = 1.0
        row[ASSETS.index(v["asset_2"])] = -1.0
        Q.append(v["view"])

    P.append(row)

P = np.asarray(P, dtype=float)
Q = np.asarray(Q, dtype=float)

if omega_method == "He-Litterman":
    Omega = he_litterman_omega(tau, P, Sigma)
else:
    Omega = np.eye(len(VIEWS)) * 0.01

views_display = pd.DataFrame(
    {
        "View": [
            "Global Architectural Coatings absolute return",
            "Global Architectural Coatings vs Industrial Coatings",
        ],
        "Q": Q,
    }
)

st.subheader("6. Investor views")

st.write(
    "Views are intentionally stored in Python only, consistent with the "
    "previous Black-Litterman project. They are assumptions for the academic "
    "model and are not presented as management guidance."
)

st.dataframe(
    views_display.style.format({"Q": "{:.2%}"}),
    hide_index=True,
    use_container_width=True,
)

# -----------------------------
# Black-Litterman posterior
# -----------------------------
posterior_mu, posterior_cov = black_litterman_posterior(
    prior_returns=Pi,
    cov=Sigma,
    tau=tau,
    P=P,
    Q=Q,
    omega=Omega,
)

posterior_df = pd.DataFrame(
    {
        "Asset sleeve": ASSETS,
        "Prior / equilibrium": Pi,
        "Posterior BL": posterior_mu,
    }
)

st.subheader("7. Black-Litterman posterior returns")

st.dataframe(
    posterior_df.style.format(
        {
            "Prior / equilibrium": "{:.2%}",
            "Posterior BL": "{:.2%}",
        }
    ),
    hide_index=True,
    use_container_width=True,
)

# -----------------------------
# Optimization
# -----------------------------
if optimization == "Maximum Sharpe":
    weights = optimize_max_sharpe(
        posterior_mu,
        posterior_cov,
        risk_free,
        min_weight,
        max_weight,
    )
elif optimization == "Minimum Volatility":
    weights = optimize_min_vol(
        posterior_mu,
        posterior_cov,
        min_weight,
        max_weight,
    )
else:
    weights = optimize_max_utility(
        posterior_mu,
        posterior_cov,
        delta,
        min_weight,
        max_weight,
    )

weights_s = pd.Series(weights, index=ASSETS)

port_ret, port_vol, port_sharpe = portfolio_stats(
    weights,
    posterior_mu,
    posterior_cov,
    risk_free,
)

st.subheader("8. Optimal portfolio")

weights_df = pd.DataFrame(
    {
        "Asset sleeve": ASSETS,
        "Optimal weight": weights,
    }
)

st.dataframe(
    weights_df.style.format({"Optimal weight": "{:.2%}"}),
    hide_index=True,
    use_container_width=True,
)

m1, m2, m3 = st.columns(3)
m1.metric("Expected return", f"{port_ret:.2%}")
m2.metric("Expected volatility", f"{port_vol:.2%}")
m3.metric("Sharpe ratio", f"{port_sharpe:.2f}")

# -----------------------------
# Frontier
# -----------------------------
st.subheader("9. Efficient frontier")

frontier_ret, frontier_vol, frontier_weights = efficient_frontier(
    posterior_mu,
    posterior_cov,
    min_weight,
    max_weight,
)

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(frontier_vol, frontier_ret, marker="o", markersize=2)
ax.scatter([port_vol], [port_ret], s=90, label="Optimal portfolio")
ax.set_xlabel("Volatility")
ax.set_ylabel("Expected return")
ax.set_title("Black-Litterman Efficient Frontier")
ax.grid(alpha=0.25)
ax.legend()
st.pyplot(fig, clear_figure=True)

# -----------------------------
# Risk diagnostics
# -----------------------------
st.subheader("10. Historical proxy risk diagnostics")

portfolio_proxy = R.values @ weights

if len(portfolio_proxy) >= 3:
    var_95 = np.percentile(portfolio_proxy, 5)
    cvar_95 = portfolio_proxy[portfolio_proxy <= var_95].mean()
else:
    var_95 = np.nan
    cvar_95 = np.nan

wealth = np.cumprod(1 + portfolio_proxy)
running_max = np.maximum.accumulate(wealth)
drawdown = wealth / running_max - 1
max_drawdown = drawdown.min()

risk_df = pd.DataFrame(
    {
        "Metric": [
            "Historical proxy VaR 95%",
            "Historical proxy CVaR 95%",
            "Maximum drawdown",
        ],
        "Value": [var_95, cvar_95, max_drawdown],
    }
)

st.dataframe(
    risk_df.style.format({"Value": "{:.2%}"}),
    hide_index=True,
    use_container_width=True,
)

st.info(
    "Because the financial-statement history is annual and short, VaR, CVaR "
    "and drawdown are illustrative diagnostics only. They should not be "
    "interpreted as market-risk estimates."
)

# -----------------------------
# Methodology summary
# -----------------------------
st.subheader("11. Methodology")

st.markdown(
    """
**Black-Litterman sequence used in this application**

1. Build historical financial return proxies from PPG segment income / assets.
2. Estimate the covariance matrix.
3. Define a reference portfolio from segment asset values.
4. Calculate equilibrium returns:
   **Π = δΣw**
5. Define investor views in Python.
6. Build He-Litterman Ω.
7. Calculate the posterior Black-Litterman expected returns.
8. Optimize the posterior portfolio subject to the selected weight limits.
9. Plot the efficient frontier.

**COMEX exposure:** Global Architectural Coatings is the sleeve containing
COMEX. PPG explicitly identifies COMEX among the primary brands of its
Architectural Coatings Latin America and Asia Pacific business.
"""
)

st.subheader("12. Sources")

st.markdown(
    """
- PPG 2025 Annual Report / Form 10-K — SEC
- PPG 2024 Annual Report / Form 10-K — SEC
- PPG 2025 full-year financial results
"""
)

st.markdown(
    """
Primary sources:
- https://www.sec.gov/Archives/edgar/data/79879/000007987926000046/ppg-20251231.htm
- https://www.sec.gov/Archives/edgar/data/79879/000007987925000034/ppg-20241231.htm
- https://investor.ppg.com/news/news-details/2026/PPG-reports-fourth-quarter-and-full-year-2025-financial-results/default.aspx
"""
)

st.caption(
    "Academic model — basic sophistication. The return series are financial "
    "proxies, not traded-asset returns. COMEX standalone financial statements "
    "are not publicly disclosed in the PPG filing at the same granularity."
)
