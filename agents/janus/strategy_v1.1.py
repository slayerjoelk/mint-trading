"""
Janus Volatility & Structure Strategy — v1.1
Updated: 2026-05-26T20:01:17.673018+00:00

Changes from previous version:
  - vol_hold_days reduced 3→1: winners resolve in avg 0.1d vs losers 0.0d. Vol fades and contango plays resolve quickly — cut hold to lock in edge.

Evidence basis:
  trades_analyzed = 1
  overall_win_rate = 1.000
  spike_fade_win_rate = 0.000 (n=0)
  buy_the_dip_win_rate = 0.000 (n=0)
  sell_the_rip_win_rate = 0.000 (n=0)
  leveraged_reversion_win_rate = 0.000 (n=0)
  contango_hedge_win_rate = 0.000 (n=0)
  avg_win_hold_days = 0.08
  avg_loss_hold_days = 0.00
"""

STRATEGY_VERSION = "1.1"

PARAMETERS = {
    "agent_id": "9c26226d-0513-4acd-83c9-8bcf4bae2dff",
    "asset_focus": "vol_products_leveraged_etfs",
    "created_at": "2026-05-02T14:38:01.216546+00:00",
    "initial_capital": 500,
    "leveraged_dev_threshold": 0.02,
    "leveraged_hold_days": 2,
    "max_positions": 5,
    "name": "Janus",
    "risk_per_trade": 0.02,
    "strategy_version": "1.0",
    "style": "volatility_structure",
    "vixy_low_threshold": 15.0,
    "vixy_spike_threshold": 0.05,
    "vol_hold_days": 1
}

TICKER_PERFORMANCE = {
    "JNJ": {
        "count": 1,
        "win_rate": 1.0,
        "avg_pnl": 23.49
    }
}

PERFORMANCE_AT_SNAPSHOT = {
    "win_rate": 1.0,
    "total_trades": 1,
    "total_pnl": 23.49,
    "avg_win": 23.49,
    "avg_loss": 0.0,
    "profit_factor": 0.0,
    "sharpe_approx": 0.0,
    "max_drawdown": 0.0,
    "current_drawdown": 0.0
}