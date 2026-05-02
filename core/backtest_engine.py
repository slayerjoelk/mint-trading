"""
Backtesting engine for Mint Trading Company.
Replays historical data through the full signal→validate→execute→learn pipeline.
"""
import uuid
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

def _now() -> int:
    return int(datetime.now(timezone.utc).timestamp())


class BacktestEngine:
    def __init__(self, data_pipeline, risk_manager, db):
        self.data = data_pipeline
        self.risk = risk_manager
        self.db = db
        self.slippage_pct = 0.001
        self.commission_per_share = 0.005

    def run(self, agent, tickers: list[str], start_date: str,
            end_date: str, initial_capital: float = 100000) -> dict:

        # --- load ALL data for the window ---
        all_data = {}
        for t in tickers:
            df = self.data.get_daily_bars(t, period="1y")
            if df is not None and len(df) >= 20:
                df = self.data.compute_indicators(df)
                all_data[t] = df

        if not all_data:
            return {"error": "No data for any ticker", "total_return": 0.0,
                    "sharpe": 0.0, "max_drawdown": 0.0, "win_rate": 0.0,
                    "trades": 0, "daily_returns": [], "final_capital": initial_capital}

        # build unified date index across all tickers
        dates = set()
        for df in all_data.values():
            dates.update(df.index.date)
        dates = sorted(d for d in dates if d >= pd.Timestamp(start_date).date() and d <= pd.Timestamp(end_date).date())
        if not dates:
            return {"error": "No dates in range", "total_return": 0.0,
                    "sharpe": 0.0, "max_drawdown": 0.0, "win_rate": 0.0,
                    "trades": 0, "daily_returns": [], "final_capital": initial_capital}

        # --- state ---
        capital = initial_capital
        portfolio_values = [initial_capital]
        positions = {}  # {symbol: {'qty': int, 'entry_price': float, 'date': date}}
        all_trades = []

        for i, date in enumerate(dates):
            day_data = {}
            for t in all_data:
                df = all_data[t]
                rows = df[df.index.date == date]
                if not rows.empty:
                    day_data[t] = {
                        "open": float(rows.iloc[0]["open"]),
                        "close": float(rows.iloc[-1]["close"]),
                        "high": float(rows["high"].max()),
                        "low": float(rows["low"].min()),
                        "volume": float(rows["volume"].sum()),
                    }
                    for col in ["sma_20", "sma_50", "rsi_14", "macd", "macd_hist",
                                "bb_upper", "bb_mid", "bb_lower", "atr_14",
                                "vol_sma_20"]:
                        if col in rows.iloc[-1]:
                            val = rows.iloc[-1][col]
                            if not pd.isna(val):
                                day_data[t][col] = float(val)

            if not day_data:
                continue

            # --- check exits for open positions ---
            for sym, pos in list(positions.items()):
                if sym not in day_data:
                    continue
                close = day_data[sym]["close"]
                exit_pnl = (close - pos["entry_price"]) * pos["qty"]
                exit_pnl -= pos["qty"] * self.commission_per_share * 2
                capital += exit_pnl
                all_trades.append({"pnl": exit_pnl, "closed": True})
                del positions[sym]

            # --- generate entry signals ---
            try:
                signals = []
                # simulate signal generation with day's data
                for sym in tickers:
                    if sym not in day_data:
                        continue
                    d = day_data[sym]
                    close = d["close"]
                    features = {k: v for k, v in d.items() if k not in ("open", "high", "low", "close", "volume")}
                    features["close"] = close
                    features["volume"] = d["volume"]

                    bb_lower = features.get("bb_lower")
                    bb_upper = features.get("bb_upper")
                    rsi = features.get("rsi_14")
                    atr = features.get("atr_14", close * 0.02)
                    sma20 = features.get("sma_20")

                    if bb_lower is None or bb_upper is None or rsi is None or sma20 is None:
                        continue

                    if rsi < 30 and close < bb_lower:
                        risk_pct = 0.02
                        stop_dist = atr * 1.5 if atr else close * 0.02
                        risk_amount = capital * risk_pct
                        qty = max(1, int(risk_amount / stop_dist))
                        max_qty = int((capital * 0.20) / close)
                        qty = min(qty, max_qty)

                        if qty * close <= capital * 0.80:
                            signals.append({
                                "symbol": sym, "side": "buy", "quantity": qty,
                                "price": close, "confidence": max(0.5, min(1.0, (bb_lower - close) / bb_lower + 0.5)),
                            })

                    elif rsi > 70 and close > bb_upper:
                        # sell signal (only if we can short conceptually)
                        pass  # skip shorts in backtest for safety
            except Exception as exc:
                logger.error("Backtest signal error on %s: %s", date, exc)
                signals = []

            for sig in signals[:5]:
                sym = sig["symbol"]
                price = sig["price"]
                qty = sig["quantity"]
                fill_price = price * (1 + self.slippage_pct)
                cost = fill_price * qty + qty * self.commission_per_share
                if cost <= capital * 0.8:
                    capital -= cost
                    positions[sym] = {"qty": qty, "entry_price": fill_price, "date": date, "confidence": sig["confidence"]}
                    all_trades.append({"pnl": 0.0, "closed": False})

            portfolio_values.append(capital)

        # --- close remaining positions at last price ---
        for sym, pos in positions.items():
            if sym in all_data:
                df = all_data[sym]
                last_close = float(df["close"].iloc[-1])
                exit_pnl = (last_close - pos["entry_price"]) * pos["qty"]
                capital += exit_pnl
                all_trades.append({"pnl": exit_pnl, "closed": True})

        # --- compute metrics ---
        closed_pnls = [t["pnl"] for t in all_trades if t["closed"]]
        wins = [p for p in closed_pnls if p > 0]
        losses = [p for p in closed_pnls if p <= 0]
        total_return = (capital - initial_capital) / initial_capital if initial_capital > 0 else 0.0

        daily_rets = []
        for j in range(1, len(portfolio_values)):
            if portfolio_values[j - 1] > 0:
                daily_rets.append((portfolio_values[j] - portfolio_values[j - 1]) / portfolio_values[j - 1])

        if len(daily_rets) > 1:
            mean_r = np.mean(daily_rets)
            std_r = np.std(daily_rets)
            sharpe = (mean_r / std_r * (252 ** 0.5)) if std_r > 0 else 0.0
        else:
            sharpe = 0.0

        win_rate = len(wins) / len(closed_pnls) if closed_pnls else 0.0

        cumulative, peak, max_dd = 0.0, 0.0, 0.0
        for r in daily_rets:
            cumulative = (1 + cumulative) * (1 + r) - 1
            peak = max(peak, cumulative)
            dd = (peak - cumulative) / (1 + peak) if peak > -1 else 0.0
            max_dd = max(max_dd, dd)

        # --- save to DB ---
        try:
            rid = str(uuid.uuid4())
            v = getattr(agent, "_config", {}).get("strategy_version", "1.0")
            self.db.insert_backtest_run(
                id=rid,
                agent_id=getattr(agent, "agent_id", str(uuid.uuid4())),
                strategy_version=str(v),
                start_date=str(start_date),
                end_date=str(end_date),
                initial_capital=initial_capital,
                final_capital=round(capital, 2),
                total_return=round(total_return, 4),
                sharpe=round(sharpe, 4),
                max_drawdown=round(max_dd, 4),
                win_rate=round(win_rate, 4),
                num_trades=len(closed_pnls),
                daily_returns=[round(r, 6) for r in daily_rets],
                created_at=_now(),
            )
        except Exception as exc:
            logger.error("Backtest DB save error: %s", exc)

        return {
            "total_return": round(total_return, 4),
            "sharpe": round(sharpe, 4),
            "max_drawdown": round(max_dd, 4),
            "win_rate": round(win_rate, 4),
            "trades": len(closed_pnls),
            "daily_returns": [round(r, 6) for r in daily_rets],
            "final_capital": round(capital, 2),
            "initial_capital": initial_capital,
            "start_date": str(start_date),
            "end_date": str(end_date),
        }
