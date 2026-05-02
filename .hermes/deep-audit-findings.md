## DEEP AUDIT — MINT TRADING COMPANY

### CRITICAL (structural — blocking hedge-fund grade)

1. **No shared learning repository.** Agents learn in isolation. Athena discovering "ranging regimes accelerate reversion" should inform Orion's regime filters and Janus's vol classification. There's no cross-agent knowledge base, no shared feature store, no ensemble voting.

2. **No backtesting framework.** Research methods are descriptive ("60% reverted in 5 days") but don't simulate actual trading with commissions, slippage, position overlap, and portfolio-level constraints. Need a backtesting engine that replays daily bars through the full signal→validate→execute→learn pipeline on historical data.

3. **No ML/statistical regime classifier.** Market regime is a hand-coded tree of SMA comparisons. Need proper regime detection: train a classifier on features (VIX level, yield curve, sector correlation, breadth indicators) against historical market states. Agents should query a shared regime model, not each run their own `get_market_regime()`.

4. **No portfolio optimization.** Croesus allocates capital by heuristic ("+20% to winner"). Need mean-variance optimization, risk parity, or max diversification weighting. Agents compete for capital but there's no mathematical basis for the allocation.

5. **No walk-forward validation.** Strategy versions are bumped without proper out-of-sample testing. Need: train on 2-year window, validate on 6-month out-of-sample, compare in-sample vs out-of-sample performance. Only deploy strategies that pass walk-forward.

6. **No execution quality metrics.** Trades are market orders with no slippage tracking. Need: VWAP comparison, implementation shortfall, fill quality by time of day, venue analysis.

7. **No alternative data.** Only price + RSS headlines. Need at minimum: options flow (put/call ratios), sector rotation signals, Fed funds futures, economic surprise index.

### HIGH (strategy depth — would transform performance)

8. **Learning loops are rigid templates.** Each agent's learn_from_trades() follows a fixed pattern. Need exploratory learning: automated hypothesis generation ("what if volume_multiplier=1.75 vs 1.5?"), parameter sweeps, confidence-interval-based rule changes instead of hard thresholds.

9. **No correlation/overlap detection.** If Athena longs AAPL and Orion also enters AAPL, there's doubled exposure with zero awareness. Need: real-time correlation matrix, position overlap alerts, concentration warnings across agents.

10. **No VaR/CVaR/tail risk.** Max drawdown is the only risk metric. Need: historical VaR, parametric VaR, expected shortfall, stress tests ("portfolio P&L if SPY -10%, -20%, -30%").

11. **Sentiment scoring is too primitive.** Keyword list with ±3 cap produces coarse sentiment. Need: FinBERT for NLP sentiment, entity extraction (who said what about which company), event classification beyond "earnings/general."

12. **No benchmark tracking.** Cannot answer: alpha over SPY? beta? information ratio? tracking error? Need: daily benchmark comparison, rolling alpha/beta, performance attribution (what drove P&L — factor exposure or skill?).

13. **Journal entries are template strings — not queryable data.** "What do all my losing trades have in common?" is unanswerable without parsing unstructured text. Need: structured trade taxonomy (regime, indicator ranges, thesis-type tags) alongside narrative journal.

14. **No ensemble/voting.** 4 agents independently score tickers. If 3 agents signal BUY on NVDA and 1 signals SELL, the system should surface the disagreement, not let each agent trade independently.

### MEDIUM (quality of life — compounding improvements)

15. **No parameter sensitivity tracking.** When volume_multiplier changes from 1.5→1.75, how much does win rate change? Need: recording sensitivity scores for each parameter change so the system learns which levers matter most.

16. **No market microstructure awareness.** All market orders, no TCA. Need: bid-ask spread estimates, time-of-day liquidity patterns, avoid trading first/last 15 minutes.

17. **No seasonal/calendar effects.** Monday effects, month-end rebalancing, opex weeks, Fed days, earnings season windows — all untracked.

18. **Data staleness monitoring.** If yfinance rate-limits and returns stale data, nobody notices. Need: freshness timestamps, staleness alerts, graceful degradation per data source.

19. **No strategy competition (paper-trade variants simultaneously).** Agents only run one strategy version at a time. A hedge fund runs v1.2 and v1.3 side-by-side on separate paper allocations and promotes the winner after sufficient data.

20. **Weekly review doesn't actually adjust capital.** Croesus prints adjustment recommendations but doesn't execute them. Need: actual capital reallocation logic that modifies agent configurations and restart parameters.
