import json
import logging
from collections import defaultdict
from datetime import datetime, timezone

import pandas as pd

from agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class OrionAgent(BaseAgent):
    """
    Momentum breakout trader on high-beta tech stocks.
    Enters on 20-day-high breakout with volume confirmation and MACD trend.
    Exits at 10-day low or after hold_days, whichever comes first.
    """

    UNIVERSE = [
        "NVDA", "AMD", "TSLA", "META", "NFLX", "CRM", "ADBE",
        "SHOP", "SNOW", "PLTR", "COIN", "SQ", "UBER", "DDOG", "NET",
    ]

    _DEFAULTS = {
        "breakout_period": 20,
        "exit_period": 10,
        "volume_multiplier": 1.5,
        "hold_days": 7,
        "strategy_version": "1.0",
    }

    # Market regimes where momentum does NOT work — skip entries
    _BAD_REGIMES = {"high_volatility", "trending_down"}

    def __init__(self, initial_capital: float, db, market, data, news, risk):
        super().__init__(
            name="Orion",
            style="momentum_breakout",
            asset_focus="high_beta_tech",
            initial_capital=initial_capital,
            db=db,
            market=market,
            data=data,
            news=news,
            risk=risk,
        )
        for k, v in self._DEFAULTS.items():
            if k not in self._config:
                self._config[k] = v
        self.save_config()

    # ------------------------------------------------------------------ research

    def research(self) -> str:
        """
        For every 20-day-high breakout with 1.5x volume confirmation, measure
        3/5/7-day forward returns, win rate, avg gain vs avg loss.
        Saves report to research_report.md.
        """
        breakout_period = int(self._config.get("breakout_period", 20))
        vol_mult = float(self._config.get("volume_multiplier", 1.5))

        lines = [
            "# Orion Research Report — Momentum Breakout Backtest",
            f"_Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}_",
            f"_Parameters: breakout_period={breakout_period}, volume_multiplier={vol_mult}x_",
            "",
            "## Per-Ticker Statistics",
            "",
        ]
        summary = []

        for ticker in self.UNIVERSE:
            try:
                df = self.data.get_daily_bars(ticker, period="1y")
                if df is None or len(df) < breakout_period + 10:
                    lines.append(f"**{ticker}**: insufficient data\n")
                    continue

                df = self.data.compute_indicators(df)
                closes = df["close"].values
                volumes = df["volume"].values
                vol_sma = df["vol_sma_20"].values
                macd_arr = df["macd"].values
                macd_hist_arr = df["macd_hist"].values

                breakout_events: dict[int, list] = {3: [], 5: [], 7: []}

                for i in range(breakout_period, len(df) - 7):
                    # 20-day high is computed on the PREVIOUS day's window to avoid lookahead
                    high_20d = max(closes[i - breakout_period:i])
                    if pd.isna(vol_sma[i]) or vol_sma[i] == 0:
                        continue

                    vol_ratio = volumes[i] / vol_sma[i]
                    breakout = closes[i] > high_20d
                    vol_confirmed = vol_ratio >= vol_mult
                    macd_positive = (not pd.isna(macd_arr[i])) and macd_arr[i] > 0
                    macd_rising = (not pd.isna(macd_hist_arr[i])) and macd_hist_arr[i] > 0

                    if not (breakout and vol_confirmed and macd_positive and macd_rising):
                        continue

                    entry_price = closes[i]
                    for h in [3, 5, 7]:
                        exit_price = closes[i + h]
                        ret = (exit_price - entry_price) / entry_price
                        breakout_events[h].append(ret)

                if not breakout_events[3]:
                    lines.append(f"**{ticker}**: no confirmed breakout events found\n")
                    continue

                def stats(rets):
                    if not rets:
                        return {"n": 0, "win_rate": 0, "avg_ret": 0, "avg_gain": 0, "avg_loss": 0}
                    wins = [r for r in rets if r > 0]
                    losses = [r for r in rets if r <= 0]
                    return {
                        "n": len(rets),
                        "win_rate": round(len(wins) / len(rets) * 100, 1),
                        "avg_ret": round(sum(rets) / len(rets) * 100, 2),
                        "avg_gain": round(sum(wins) / len(wins) * 100, 2) if wins else 0,
                        "avg_loss": round(sum(losses) / len(losses) * 100, 2) if losses else 0,
                    }

                lines.append(f"### {ticker}")
                lines.append("| Horizon | N Events | Win Rate | Avg Return | Avg Gain | Avg Loss |")
                lines.append("|---------|----------|----------|------------|----------|----------|")
                for h in [3, 5, 7]:
                    s = stats(breakout_events[h])
                    lines.append(
                        f"| {h}d | {s['n']} | {s['win_rate']}% | {s['avg_ret']:+.2f}% "
                        f"| {s['avg_gain']:+.2f}% | {s['avg_loss']:+.2f}% |"
                    )
                lines.append("")

                s7 = stats(breakout_events[7])
                summary.append({
                    "ticker": ticker,
                    "n": s7["n"],
                    "win_rate_7d": s7["win_rate"],
                    "avg_ret_7d": s7["avg_ret"],
                    "avg_gain_7d": s7["avg_gain"],
                    "avg_loss_7d": s7["avg_loss"],
                })

            except Exception as exc:
                logger.error("Orion research [%s]: %s", ticker, exc)
                lines.append(f"**{ticker}**: error — {exc}\n")

        if summary:
            lines += [
                "## Universe Summary (sorted by 7-day win rate)",
                "",
                "| Ticker | Breakouts | Win% | Avg Return | Avg Gain | Avg Loss |",
                "|--------|-----------|------|------------|----------|----------|",
            ]
            for r in sorted(summary, key=lambda x: x["win_rate_7d"], reverse=True):
                lines.append(
                    f"| {r['ticker']} | {r['n']} | {r['win_rate_7d']}% "
                    f"| {r['avg_ret_7d']:+.2f}% | {r['avg_gain_7d']:+.2f}% "
                    f"| {r['avg_loss_7d']:+.2f}% |"
                )

            avg_wr = sum(r["win_rate_7d"] for r in summary) / len(summary)
            best = max(summary, key=lambda r: r["win_rate_7d"])
            lines += [
                "",
                "## Conclusions",
                "",
                f"- Universe avg 7-day breakout win rate: **{avg_wr:.1f}%**",
                f"- Best momentum ticker: **{best['ticker']}** "
                f"({best['win_rate_7d']}% win rate, {best['avg_gain_7d']:+.2f}% avg gain)",
                f"- Strategy viability: **{'Strong' if avg_wr > 55 else 'Moderate' if avg_wr > 45 else 'Weak'}**",
                "- Note: backtest does not account for slippage on breakout entry or regime filtering.",
            ]

        path = self._agent_dir / "research_report.md"
        path.write_text("\n".join(lines))
        logger.info("Orion research report saved → %s", path)
        return str(path)

    # ------------------------------------------------------------------ signals

    def generate_signals(self) -> list[dict]:
        """
        Two signal types:
        1. EXIT signals for open positions that hit 10-day low or 7-day hold.
        2. ENTRY signals for new breakouts with volume + MACD confirmation.
        Exits are prioritised; combined list capped at 5.
        """
        breakout_period = int(self._config.get("breakout_period", 20))
        exit_period = int(self._config.get("exit_period", 10))
        vol_mult = float(self._config.get("volume_multiplier", 1.5))
        hold_days = int(self._config.get("hold_days", 7))
        regime = self.data.get_market_regime()

        exit_signals = []
        entry_signals = []

        # --- EXIT signals: check open positions ---
        try:
            conn = self.db._conn()
            open_trades = conn.execute(
                "SELECT ticker, entry_price, opened_at, qty FROM trades "
                "WHERE agent_id = ? AND status = 'open'",
                (self.agent_id,),
            ).fetchall()
            conn.close()
        except Exception as exc:
            logger.error("Orion signal DB query error: %s", exc)
            open_trades = []

        now_ts = int(datetime.now(timezone.utc).timestamp())

        for trade in open_trades:
            ticker = trade["ticker"]
            entry_price = float(trade["entry_price"] or 0)
            opened_at = trade["opened_at"] or now_ts
            held_days = (now_ts - opened_at) / 86400

            try:
                df = self.data.get_daily_bars(ticker, period="1mo")
                if df is None or len(df) < exit_period:
                    continue
                df = self.data.compute_indicators(df)
                current_close = float(df["close"].iloc[-1])
                low_nd = float(df["low"].tail(exit_period).min())
                atr = float(df["atr_14"].iloc[-1]) if not pd.isna(df["atr_14"].iloc[-1]) else 0

                triggered_7d = held_days >= hold_days
                triggered_low = current_close < low_nd

                if triggered_7d or triggered_low:
                    trigger_reason = []
                    if triggered_7d:
                        trigger_reason.append(f"max hold reached ({held_days:.1f}d ≥ {hold_days}d)")
                    if triggered_low:
                        pct_drop = (current_close - low_nd) / entry_price * 100
                        trigger_reason.append(
                            f"price ${current_close:.2f} broke {exit_period}-day low "
                            f"${low_nd:.2f} (entry=${entry_price:.2f}, "
                            f"draw={pct_drop:.2f}%)"
                        )

                    qty = self.risk.calculate_position_size_atr(
                        agent_capital=self.capital,
                        price=current_close,
                        atr=atr if atr > 0 else current_close * 0.02,
                        risk_per_trade=float(self._config.get("risk_per_trade", 0.02)),
                    )
                    qty = max(1, qty)

                    reason = (
                        f"ORION_MOMENTUM_EXIT | {ticker} @ ${current_close:.2f} | "
                        f"Entry=${entry_price:.2f} | Held={held_days:.1f}d | "
                        f"{exit_period}d_low=${low_nd:.2f} | "
                        f"Regime_at_exit={regime} | "
                        f"Trigger: {'; '.join(trigger_reason)}"
                    )
                    exit_signals.append({
                        "symbol": ticker,
                        "side": "sell",
                        "quantity": qty,
                        "type": "market",
                        "reason": reason,
                        "confidence": 1.0,
                    })
            except Exception as exc:
                logger.error("Orion exit check [%s]: %s", ticker, exc)

        # --- ENTRY signals: check regime first ---
        if regime not in self._BAD_REGIMES:
            for ticker in self.UNIVERSE:
                if len(exit_signals) + len(entry_signals) >= 5:
                    break
                try:
                    df = self.data.get_daily_bars(ticker, period="3mo")
                    if df is None or len(df) < breakout_period + 2:
                        continue
                    df = self.data.compute_indicators(df)

                    latest = df.iloc[-1]
                    prev = df.iloc[-2]

                    close = float(latest["close"])
                    volume = float(latest["volume"])
                    vol_sma_20 = float(latest["vol_sma_20"]) if not pd.isna(latest["vol_sma_20"]) else None
                    macd = float(latest["macd"]) if not pd.isna(latest["macd"]) else None
                    macd_hist = float(latest["macd_hist"]) if not pd.isna(latest["macd_hist"]) else None
                    atr = float(latest["atr_14"]) if not pd.isna(latest["atr_14"]) else close * 0.02
                    rsi = float(latest["rsi_14"]) if not pd.isna(latest["rsi_14"]) else None

                    if vol_sma_20 is None or vol_sma_20 == 0:
                        continue

                    # 20-day high is the max close BEFORE today (iloc[-2] window)
                    high_20d_prev = float(df["close"].iloc[-(breakout_period + 1):-1].max())
                    vol_ratio = volume / vol_sma_20

                    breakout = close > high_20d_prev
                    vol_confirmed = vol_ratio >= vol_mult
                    macd_positive = macd is not None and macd > 0
                    macd_rising = macd_hist is not None and macd_hist > 0

                    if not (breakout and vol_confirmed and macd_positive and macd_rising):
                        continue

                    # Confidence: volume strength + MACD magnitude
                    vol_score = min(1.0, (vol_ratio - vol_mult) / vol_mult * 0.5 + 0.5)
                    macd_score = min(1.0, abs(macd_hist) / (abs(macd) + 1e-9) * 0.5 + 0.5) if macd else 0.5
                    confidence = round((vol_score + macd_score) / 2, 4)

                    qty = self.risk.calculate_position_size_atr(
                        agent_capital=self.capital,
                        price=close,
                        atr=atr,
                        risk_per_trade=float(self._config.get("risk_per_trade", 0.02)),
                    )
                    if qty <= 0:
                        continue

                    reason = (
                        f"ORION_MOMENTUM_BREAKOUT_BUY | {ticker} @ ${close:.2f} | "
                        f"20d_high=${high_20d_prev:.2f} (broke out by ${close - high_20d_prev:.2f}) | "
                        f"Volume={volume:,.0f} ({vol_ratio:.2f}x avg, threshold={vol_mult}x) | "
                        f"MACD={macd:.4f} MACD_hist={macd_hist:.4f} (positive & rising) | "
                        f"RSI={rsi:.1f if rsi else 'N/A'} | ATR=${atr:.2f} | "
                        f"Regime_at_entry={regime} | "
                        f"Exit plan: sell if price < {exit_period}d low OR after {hold_days}d"
                    )

                    entry_signals.append({
                        "symbol": ticker,
                        "side": "buy",
                        "quantity": qty,
                        "type": "market",
                        "reason": reason,
                        "confidence": confidence,
                    })

                except Exception as exc:
                    logger.error("Orion entry check [%s]: %s", ticker, exc)
        else:
            logger.info(
                "Orion skipping entries — regime '%s' unfavorable for momentum", regime
            )

        entry_signals.sort(key=lambda s: s["confidence"], reverse=True)
        combined = exit_signals + entry_signals
        return combined[:5]

    # ------------------------------------------------------------------ learning

    def learn_from_trades(self):
        """
        Analyze closed trades. Test whether RSI filter would improve win rate.
        Check if 4-day exit beats 7-day. Find optimal volume multiplier. Save version.
        """
        try:
            conn = self.db._conn()
            rows = conn.execute(
                """
                SELECT t.ticker, t.side, t.entry_price, t.exit_price, t.pnl,
                       t.opened_at, t.closed_at,
                       j.market_conditions, j.pre_trade_reasoning
                FROM trades t
                LEFT JOIN trade_journal j ON j.trade_id = t.id
                WHERE t.agent_id = ? AND t.status = 'closed'
                ORDER BY t.closed_at DESC LIMIT 60
                """,
                (self.agent_id,),
            ).fetchall()
            conn.close()
        except Exception as exc:
            logger.error("Orion learn_from_trades DB error: %s", exc)
            return

        if not rows:
            logger.info("Orion: no closed trades to learn from yet")
            return

        trades = []
        for row in rows:
            mc = {}
            if row["market_conditions"]:
                try:
                    mc = json.loads(row["market_conditions"])
                except Exception:
                    pass
            hold_secs = (row["closed_at"] or 0) - (row["opened_at"] or 0)
            pnl = float(row["pnl"] or 0)
            trades.append({
                "ticker": row["ticker"],
                "pnl": pnl,
                "won": pnl > 0,
                "regime": mc.get("regime", "unknown"),
                "rsi": mc.get("rsi_14"),
                "vol_ratio": mc.get("vol_ratio"),
                "hold_days": hold_secs / 86400 if hold_secs > 0 else 0,
                "pre_reasoning": row["pre_trade_reasoning"] or "",
            })

        def _wr(items):
            return sum(1 for t in items if t["won"]) / len(items) if items else 0.0

        overall_wr = _wr(trades)

        # Regime breakdown
        regime_groups: dict = defaultdict(list)
        for t in trades:
            regime_groups[t["regime"]].append(t)
        regime_stats = {
            r: {"count": len(g), "win_rate": round(_wr(g), 3)}
            for r, g in regime_groups.items()
        }

        # Ticker breakdown
        ticker_groups: dict = defaultdict(list)
        for t in trades:
            ticker_groups[t["ticker"]].append(t)
        ticker_stats = {
            tk: {
                "count": len(g),
                "win_rate": round(_wr(g), 3),
                "avg_pnl": round(sum(t["pnl"] for t in g) / len(g), 2),
            }
            for tk, g in ticker_groups.items()
        }

        # Hold-day analysis: bucket into early (<4d) and late (4-7d) exits
        early_exits = [t for t in trades if t["hold_days"] < 4]
        late_exits = [t for t in trades if t["hold_days"] >= 4]
        early_wr = _wr(early_exits)
        late_wr = _wr(late_exits)

        # Volume ratio analysis: did higher vol_ratio entries perform better?
        high_vol = [t for t in trades if t["vol_ratio"] and t["vol_ratio"] > 2.0]
        med_vol = [t for t in trades if t["vol_ratio"] and 1.5 <= t["vol_ratio"] <= 2.0]
        high_vol_wr = _wr(high_vol)
        med_vol_wr = _wr(med_vol)

        changes = []

        # If early exits significantly outperform late exits, reduce hold_days
        if len(early_exits) >= 5 and len(late_exits) >= 5 and early_wr > late_wr + 0.15:
            old_hold = int(self._config.get("hold_days", 7))
            new_hold = max(3, old_hold - 2)
            if new_hold != old_hold:
                self._config["hold_days"] = new_hold
                changes.append(
                    f"hold_days reduced {old_hold}→{new_hold}: "
                    f"early exits (<4d) win_rate={early_wr:.1%} vs "
                    f"late exits (4-7d) win_rate={late_wr:.1%}. "
                    "Momentum exhausts faster than expected — locking in earlier."
                )

        # If high-volume breakouts significantly outperform median-volume, raise the threshold
        if len(high_vol) >= 5 and len(med_vol) >= 5 and high_vol_wr > med_vol_wr + 0.15:
            old_mult = float(self._config.get("volume_multiplier", 1.5))
            new_mult = round(min(2.5, old_mult + 0.25), 2)
            if new_mult != old_mult:
                self._config["volume_multiplier"] = new_mult
                changes.append(
                    f"volume_multiplier raised {old_mult}→{new_mult}: "
                    f">2x vol breakouts win_rate={high_vol_wr:.1%} vs "
                    f"1.5-2x vol breakouts win_rate={med_vol_wr:.1%}. "
                    "Higher volume confirmation produces cleaner momentum signals."
                )

        # If win rate is below 40% across enough trades, tighten regime filter
        if len(trades) >= 15 and overall_wr < 0.40:
            # Already filtering trending_down and high_volatility — no more to add safely
            # Instead reduce ATR size to only hold positions when not too volatile
            old_hold = int(self._config.get("hold_days", 7))
            if old_hold > 4:
                self._config["hold_days"] = 4
                changes.append(
                    f"hold_days emergency reduction {old_hold}→4: "
                    f"overall win rate {overall_wr:.1%} below 40% across {len(trades)} trades. "
                    "Cutting holds early to limit loss exposure until win rate recovers."
                )

        # --- version & save ---
        current_v = str(self._config.get("strategy_version", "1.0"))
        try:
            major, minor = current_v.rsplit(".", 1)
            new_v = f"{major}.{int(minor) + 1}"
        except Exception:
            new_v = "1.1"

        best_ticker = max(ticker_stats, key=lambda tk: ticker_stats[tk]["win_rate"], default="N/A") if ticker_stats else "N/A"
        best_regime = max(regime_stats, key=lambda r: regime_stats[r]["win_rate"], default="unknown")

        description = (
            f"v{new_v} | trades={len(trades)} wr={overall_wr:.1%} "
            f"best_ticker={best_ticker} best_regime={best_regime} "
            f"early_exit_wr={early_wr:.1%} late_exit_wr={late_wr:.1%} | "
            + ("; ".join(changes) if changes else "no parameter changes — monitoring")
        )

        change_block = "\n".join(f"  - {c}" for c in changes) if changes else "  - None (monitoring phase)"
        code_lines = [
            '"""',
            f"Orion Momentum Breakout Strategy — v{new_v}",
            f"Updated: {datetime.now(timezone.utc).isoformat()}",
            "",
            "Changes from previous version:",
            change_block,
            "",
            "Evidence basis:",
            f"  trades_analyzed = {len(trades)}",
            f"  overall_win_rate = {overall_wr:.3f}",
            f"  early_exit_win_rate (<4d) = {early_wr:.3f} (n={len(early_exits)})",
            f"  late_exit_win_rate (4-7d) = {late_wr:.3f} (n={len(late_exits)})",
            f"  high_vol_breakout_wr (>2x) = {high_vol_wr:.3f} (n={len(high_vol)})",
            f"  med_vol_breakout_wr (1.5-2x) = {med_vol_wr:.3f} (n={len(med_vol)})",
            f"  best_ticker = {best_ticker}",
            f"  best_regime = {best_regime}",
            '"""',
            "",
            f'STRATEGY_VERSION = "{new_v}"',
            "",
            "PARAMETERS = " + json.dumps(self._config, indent=4, default=str),
            "",
            "REGIME_PERFORMANCE = " + json.dumps(regime_stats, indent=4),
            "",
            "TICKER_PERFORMANCE = " + json.dumps(ticker_stats, indent=4),
            "",
            "PERFORMANCE_AT_SNAPSHOT = " + json.dumps(self.get_performance_metrics(), indent=4),
        ]
        code_str = "\n".join(code_lines)

        self._config["strategy_version"] = new_v
        self.save_config()
        self.save_strategy(version=new_v, description=description, code_str=code_str)
        logger.info("Orion updated → v%s | %s", new_v, description[:140])
