# COMEX / PPG — Black-Litterman Asset Optimization

Basic academic Black-Litterman model using public PPG financial-statement data.

## Key methodological point

COMEX is a primary brand within PPG's Global Architectural Coatings segment.
PPG does not disclose a complete standalone Comex portfolio/balance sheet in
its public Form 10-K at the level needed for a conventional market-return
Black-Litterman model.

Therefore, the model uses PPG's three reportable operating segments as
asset sleeves:

1. Global Architectural Coatings — contains COMEX exposure
2. Performance Coatings
3. Industrial Coatings

The financial return proxy is:

    Segment income / Segment assets

This is explicitly a proxy and not a stock-market return.

## Black-Litterman components

- Reference portfolio: asset-value weighted or equal weight
- Risk-free rate: user input
- Risk aversion: market implied or fixed
- Covariance: Ledoit-Wolf or sample
- Tau: bootstrap/statistical or fixed
- Omega: He-Litterman or identity
- Investor views: defined directly in Python
- Optimization: Maximum Sharpe, Minimum Volatility, Maximum Utility
- Weight limits and short selling
- Efficient frontier
- Historical proxy VaR / CVaR / drawdown

## Sources

PPG 2025 Form 10-K:
https://www.sec.gov/Archives/edgar/data/79879/000007987926000046/ppg-20251231.htm

PPG 2024 Form 10-K:
https://www.sec.gov/Archives/edgar/data/79879/000007987925000034/ppg-20241231.htm

PPG 2025 full-year results:
https://investor.ppg.com/news/news-details/2026/PPG-reports-fourth-quarter-and-full-year-2025-financial-results/default.aspx
