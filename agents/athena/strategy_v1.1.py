"""
Athena Mean Reversion Strategy — v1.1
Updated: 2026-05-26T20:01:17.664193+00:00

Changes from previous version:
  - hold_days reduced 5→2. Winners resolving in avg 0.1d vs losers 0.0d. Tightening hold to lock in gains before mean-reversion exhausts.

Evidence basis:
  trades_analyzed = 1
  overall_win_rate = 1.000
  best_regime = unknown (win_rate=1.0)
  worst_regime = unknown (win_rate=1.0)
  best_ticker = AAPL
  avg_win_hold_days = 0.08
  avg_loss_hold_days = 0.00
"""

STRATEGY_VERSION = "1.1"

PARAMETERS = {
    "agent_id": "3035fee6-23a4-4497-b31f-26f5bc1cdd4f",
    "asset_focus": "large_cap_equities",
    "bb_std_dev": 2.0,
    "created_at": "2026-05-02T14:38:01.211717+00:00",
    "hold_days": 2,
    "initial_capital": 500,
    "lookback_period": 20,
    "max_atr_pct": 0.03,
    "max_positions": 5,
    "name": "Athena",
    "risk_per_trade": 0.02,
    "rsi_overbought": 70,
    "rsi_oversold": 30,
    "strategy_version": "1.0",
    "style": "mean_reversion"
}

REGIME_PERFORMANCE = {
    "unknown": {
        "count": 1,
        "win_rate": 1.0,
        "avg_pnl": 214.12
    }
}

TICKER_PERFORMANCE = {
    "AAPL": {
        "count": 1,
        "win_rate": 1.0,
        "avg_pnl": 214.12
    }
}

PERFORMANCE_AT_SNAPSHOT = {
    "win_rate": 1.0,
    "total_trades": 1,
    "total_pnl": 214.12,
    "avg_win": 214.12,
    "avg_loss": 0.0,
    "profit_factor": 0.0,
    "sharpe_approx": 0.0,
    "max_drawdown": 0.0,
    "current_drawdown": 0.0
}