"""
Orion Momentum Breakout Strategy — v1.1
Updated: 2026-05-26T20:01:17.667964+00:00

Changes from previous version:
  - None (monitoring phase)

Evidence basis:
  trades_analyzed = 1
  overall_win_rate = 1.000
  early_exit_win_rate (<4d) = 1.000 (n=1)
  late_exit_win_rate (4-7d) = 0.000 (n=0)
  high_vol_breakout_wr (>2x) = 0.000 (n=0)
  med_vol_breakout_wr (1.5-2x) = 0.000 (n=0)
  best_ticker = NVDA
  best_regime = unknown
"""

STRATEGY_VERSION = "1.1"

PARAMETERS = {
    "agent_id": "3ae5d7e6-8b0d-40a6-b088-08c91f3818af",
    "asset_focus": "high_beta_tech",
    "breakout_period": 20,
    "created_at": "2026-05-02T14:38:01.213662+00:00",
    "exit_period": 10,
    "hold_days": 7,
    "initial_capital": 500,
    "max_positions": 5,
    "name": "Orion",
    "risk_per_trade": 0.02,
    "strategy_version": "1.0",
    "style": "momentum_breakout",
    "volume_multiplier": 1.5
}

REGIME_PERFORMANCE = {
    "unknown": {
        "count": 1,
        "win_rate": 1.0
    }
}

TICKER_PERFORMANCE = {
    "NVDA": {
        "count": 1,
        "win_rate": 1.0,
        "avg_pnl": 45.86
    }
}

PERFORMANCE_AT_SNAPSHOT = {
    "win_rate": 1.0,
    "total_trades": 1,
    "total_pnl": 45.86,
    "avg_win": 45.86,
    "avg_loss": 0.0,
    "profit_factor": 0.0,
    "sharpe_approx": 0.0,
    "max_drawdown": 0.0,
    "current_drawdown": 0.0
}