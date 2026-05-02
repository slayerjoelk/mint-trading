import json
import logging
from collections import defaultdict
from datetime import datetime, timezone

import pandas as pd

from agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class AthenaAgent(BaseAgent):
    """
    Mean reversion trader on large-cap equities.
    Enters when price deviates 2σ from 20-day SMA with RSI confirmation.
    Exits at SMA or after hold_days.
    """

    UNIVERSE = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "JPM", "V",
        "JNJ", "PG", "WMT", "DIS", "NFLX", "ADBE", "CRM",
    ]

    _DEFAULTS = {
        "lookback_period": 20,
        "bb_std_dev": 2.0,
        "rsi_oversold": 30,
        "rsi_overbought": 70,
        "max_atr_pct": 0.03,
        "hold_days": 5,
        "strategy_version": "1.0",
    }

    def __init__(self, initial_capital: float, db, market, data, news, risk):
        super().__init__(
            name="Athena",
            style="mean_reversion",
            asset_focus="large_cap_equities",
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
        Pull 1 year of daily bars for each ticker. For every close that is
        >=2σ from the 20-day SMA, measure what % revert to mean within 1/3/5 days
        vs what % continue trending. Saves findings to research_report.md.
        """
        lookback = int(self._config.get("lookback_period", 20))
        std_mult = float(self._config.get("bb_std_dev", 2.0))
        lines = [
            "# Athena Research Report — Mean Reversion Backtest",
            f"_Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}_",
            f"_Parameters: lookback={lookback}, BB_std_dev={std_mult}_",
            "",
            "## Per-Ticker Statistics",
            "",
        ]
        summary = []

        for ticker in self.UNIVERSE:
            try:
                df = self.data.get_daily_bars(ticker, period="1y")
                if df is None or len(df) < lookback + 10:
                    lines.append(f"**{ticker}**: insufficient data\n")
                    continue

                df = self.data.compute_indicators(df)
                closes = df["close"].values
                sma_arr = df["sma_20"].values
                roll_std = df["close"].rolling(lookback).std()

                # key: horizon → list of (direction, reverted_bool)
                events: dict[int, list] = {1: [], 3: [], 5: []}

                for i in range(lookback, len(df) - 5):
                    s = roll_std.iloc[i]
                    if pd.isna(s) or s == 0:
                        continue
                    z = (closes[i] - sma_arr[i]) / s
                    if abs(z) < std_mult:
                        continue
                    direction = "below" if z < 0 else "above"
                    for h in [1, 3, 5]:
                        future_sma = sma_arr[i + h]
                        future_close = closes[i + h]
                        if pd.isna(future_sma):
                            continue
                        reverted = (future_close > future_sma) if direction == "below" else (future_close < future_sma)
                        events[h].append((direction, reverted, abs(z)))

                if not events[1]:
                    lines.append(f"**{ticker}**: no {std_mult}σ events in 1-year window\n")
                    continue

                def rev_rate(h, direction):
                    subset = [r for d, r, _ in events[h] if d == direction]
                    return (sum(subset) / len(subset) * 100) if subset else 0.0

                def avg_z(direction):
                    zs = [z for d, _, z in events[1] if d == direction]
                    return (sum(zs) / len(zs)) if zs else 0.0

                buy_n = sum(1 for d, _, _ in events[1] if d == "below")
                sell_n = sum(1 for d, _, _ in events[1] if d == "above")

                lines.append(f"### {ticker}")
                lines.append(
                    f"- 2σ below events (buy signals): **{buy_n}** "
                    f"| avg deviation: {avg_z('below'):.2f}σ"
                )
                lines.append(
                    f"- 2σ above events (sell signals): **{sell_n}** "
                    f"| avg deviation: {avg_z('above'):.2f}σ"
                )
                lines.append("")
                lines.append("| Horizon | Buy → Reverted % | Sell → Reverted % |")
                lines.append("|---------|------------------|-------------------|")
                for h in [1, 3, 5]:
                    lines.append(
                        f"| {h}d | {rev_rate(h, 'below'):.1f}% | {rev_rate(h, 'above'):.1f}% |"
                    )
                lines.append("")

                summary.append({
                    "ticker": ticker,
                    "buy_n": buy_n,
                    "sell_n": sell_n,
                    "rev1d_buy": rev_rate(1, "below"),
                    "rev3d_buy": rev_rate(3, "below"),
                    "rev5d_buy": rev_rate(5, "below"),
                    "rev5d_sell": rev_rate(5, "above"),
                })

            except Exception as exc:
                logger.error("Athena research [%s]: %s", ticker, exc)
                lines.append(f"**{ticker}**: error — {exc}\n")

        if summary:
            lines += [
                "## Universe Summary (sorted by 5-day buy reversion rate)",
                "",
                "| Ticker | Buy Signals | Sell Signals | 1d Rev% | 3d Rev% | 5d Rev% | 5d Sell Rev% |",
                "|--------|-------------|--------------|---------|---------|---------|--------------|",
            ]
            for r in sorted(summary, key=lambda x: x["rev5d_buy"], reverse=True):
                lines.append(
                    f"| {r['ticker']} | {r['buy_n']} | {r['sell_n']} "
                    f"| {r['rev1d_buy']:.1f}% | {r['rev3d_buy']:.1f}% "
                    f"| {r['rev5d_buy']:.1f}% | {r['rev5d_sell']:.1f}% |"
                )

            avg_buy = sum(r["rev5d_buy"] for r in summary) / len(summary)
            best = max(summary, key=lambda r: r["rev5d_buy"])
            viability = "Strong" if avg_buy > 60 else "Moderate" if avg_buy > 45 else "Weak"

            lines += [
                "",
                "## Conclusions",
                "",
                f"- Universe avg 5-day buy reversion rate: **{avg_buy:.1f}%**",
                f"- Best mean-reversion ticker: **{best['ticker']}** "
                f"({best['rev5d_buy']:.1f}% 5d reversion, {best['buy_n']} events)",
                f"- Strategy viability: **{viability}**",
                f"- Recommended lookback: {lookback} days (current) — "
                "consider 15 if fast reversion > slow reversion across most tickers",
                "",
                "## Methodology",
                "At each bar where |z-score| ≥ 2σ, a signal is counted. "
                "'Reverted' means price crossed back above/below the 20-day SMA "
                "within the given horizon. Continue-trending events are the complement.",
            ]

        path = self._agent_dir / "research_report.md"
        path.write_text("\n".join(lines))
        logger.info("Athena research report saved → %s", path)
        return str(path)

    # ------------------------------------------------------------------ signals

    def generate_signals(self) -> list[dict]:
        """
        Scan universe for Bollinger Band + RSI confluences.
        Returns up to 5 signals ranked by confidence (distance from mean).
        """
        rsi_oversold = int(self._config.get("rsi_oversold", 30))
        rsi_overbought = int(self._config.get("rsi_overbought", 70))
        max_atr_pct = float(self._config.get("max_atr_pct", 0.03))
        hold_days = int(self._config.get("hold_days", 5))
        regime = self.data.get_market_regime()
        signals = []

        for ticker in self.UNIVERSE:
            try:
                f = self.data.get_features(ticker)
                if not f:
                    continue

                close = f.get("close")
                bb_lower = f.get("bb_lower")
                bb_upper = f.get("bb_upper")
                bb_mid = f.get("bb_mid")
                rsi = f.get("rsi_14")
                atr = f.get("atr_14")
                sma20 = f.get("sma_20")
                macd_hist = f.get("macd_hist")
                vol_ratio = f.get("vol_ratio")

                if any(v is None for v in [close, bb_lower, bb_upper, bb_mid, rsi, atr, sma20]):
                    continue
                if close <= 0:
                    continue

                # ATR/price filter — skip high-volatility names unsuitable for mean reversion
                atr_pct = atr / close
                if atr_pct > max_atr_pct:
                    continue

                half_band = bb_mid - bb_lower
                if half_band <= 0:
                    continue

                side = None
                confidence = 0.0
                dev_desc = ""
                entry_thesis = ""

                if close < bb_lower and rsi < rsi_oversold:
                    side = "buy"
                    # Confidence: how many half-bands below lower BB
                    deviation_multiples = (bb_lower - close) / half_band
                    confidence = min(1.0, 0.5 + deviation_multiples * 0.5)
                    pct_below = (bb_lower - close) / close * 100
                    dev_desc = f"${(bb_lower - close):.2f} ({pct_below:.2f}%) below BB_lower"
                    entry_thesis = (
                        f"Price {dev_desc}. RSI={rsi:.1f} confirms oversold (<{rsi_oversold}). "
                        f"ATR={atr:.2f} ({atr_pct*100:.2f}% of price — within {max_atr_pct*100:.0f}% threshold). "
                        f"Target reversion to SMA20=${sma20:.2f} within {hold_days} days. "
                        f"MACD_hist={macd_hist:.4f if macd_hist else 'N/A'}. "
                        f"Vol_ratio={vol_ratio:.2f if vol_ratio else 'N/A'}. "
                        f"Market regime: {regime}."
                    )

                elif close > bb_upper and rsi > rsi_overbought:
                    side = "sell"
                    deviation_multiples = (close - bb_upper) / half_band
                    confidence = min(1.0, 0.5 + deviation_multiples * 0.5)
                    pct_above = (close - bb_upper) / close * 100
                    dev_desc = f"${(close - bb_upper):.2f} ({pct_above:.2f}%) above BB_upper"
                    entry_thesis = (
                        f"Price {dev_desc}. RSI={rsi:.1f} confirms overbought (>{rsi_overbought}). "
                        f"ATR={atr:.2f} ({atr_pct*100:.2f}% of price — within threshold). "
                        f"Target reversion to SMA20=${sma20:.2f} within {hold_days} days. "
                        f"MACD_hist={macd_hist:.4f if macd_hist else 'N/A'}. "
                        f"Vol_ratio={vol_ratio:.2f if vol_ratio else 'N/A'}. "
                        f"Market regime: {regime}."
                    )

                else:
                    continue

                qty = self.risk.calculate_position_size_atr(
                    agent_capital=self.capital,
                    price=close,
                    atr=atr,
                    risk_per_trade=float(self._config.get("risk_per_trade", 0.02)),
                )
                if qty <= 0:
                    continue

                reason = (
                    f"ATHENA_MEAN_REVERSION_{side.upper()} | {ticker} @ ${close:.2f} | "
                    f"BB[{bb_lower:.2f} / {bb_mid:.2f} / {bb_upper:.2f}] | "
                    f"RSI={rsi:.1f} | ATR=${atr:.2f} ({atr_pct*100:.2f}%) | "
                    f"Regime={regime} | {entry_thesis}"
                )

                signals.append({
                    "symbol": ticker,
                    "side": side,
                    "quantity": qty,
                    "type": "market",
                    "reason": reason,
                    "confidence": round(confidence, 4),
                })

            except Exception as exc:
                logger.error("Athena signal [%s]: %s", ticker, exc)

        signals.sort(key=lambda s: s["confidence"], reverse=True)
        return signals[:5]

    # ------------------------------------------------------------------ learning

    def learn_from_trades(self):
        """
        Analyze closed trades. Identify which regimes and tickers reverted fastest.
        Update RSI thresholds or ATR filter if evidence warrants. Save strategy version.
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
            logger.error("Athena learn_from_trades DB error: %s", exc)
            return

        if not rows:
            logger.info("Athena: no closed trades to learn from yet")
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
            trades.append({
                "ticker": row["ticker"],
                "pnl": float(row["pnl"] or 0),
                "regime": mc.get("regime", "unknown"),
                "rsi": mc.get("rsi_14"),
                "hold_days": hold_secs / 86400 if hold_secs > 0 else 0,
                "vol_ratio": mc.get("vol_ratio"),
            })

        # --- regime analysis ---
        regime_pnls: dict = defaultdict(list)
        ticker_pnls: dict = defaultdict(list)
        for t in trades:
            regime_pnls[t["regime"]].append(t["pnl"])
            ticker_pnls[t["ticker"]].append(t["pnl"])

        def _wr(pnls):
            return sum(1 for p in pnls if p > 0) / len(pnls) if pnls else 0.0

        regime_stats = {
            r: {
                "count": len(p),
                "win_rate": round(_wr(p), 3),
                "avg_pnl": round(sum(p) / len(p), 2),
            }
            for r, p in regime_pnls.items()
        }
        ticker_stats = {
            tk: {
                "count": len(p),
                "win_rate": round(_wr(p), 3),
                "avg_pnl": round(sum(p) / len(p), 2),
            }
            for tk, p in ticker_pnls.items()
        }

        all_pnls = [t["pnl"] for t in trades]
        overall_wr = _wr(all_pnls)
        best_regime = max(regime_stats, key=lambda r: regime_stats[r]["win_rate"], default="unknown")
        worst_regime = min(regime_stats, key=lambda r: regime_stats[r]["win_rate"], default="unknown")
        best_ticker = max(ticker_stats, key=lambda tk: ticker_stats[tk]["win_rate"], default="N/A") if ticker_stats else "N/A"

        # Average hold duration for winners vs losers
        win_holds = [t["hold_days"] for t in trades if t["pnl"] > 0 and t["hold_days"] > 0]
        loss_holds = [t["hold_days"] for t in trades if t["pnl"] <= 0 and t["hold_days"] > 0]
        avg_win_hold = sum(win_holds) / len(win_holds) if win_holds else 0
        avg_loss_hold = sum(loss_holds) / len(loss_holds) if loss_holds else 0

        changes = []

        # If ranging regime significantly outperforms trending, tighten RSI to filter trending entries
        ranging_wr = regime_stats.get("ranging", {}).get("win_rate", 0.5)
        trending_up_wr = regime_stats.get("trending_up", {}).get("win_rate", 0.5)
        trending_down_wr = regime_stats.get("trending_down", {}).get("win_rate", 0.5)

        if (
            len(trades) >= 10
            and ranging_wr > 0.60
            and (trending_up_wr < 0.45 or trending_down_wr < 0.45)
        ):
            old_os = int(self._config.get("rsi_oversold", 30))
            old_ob = int(self._config.get("rsi_overbought", 70))
            new_os = max(25, old_os - 2)
            new_ob = min(75, old_ob + 2)
            if new_os != old_os or new_ob != old_ob:
                self._config["rsi_oversold"] = new_os
                self._config["rsi_overbought"] = new_ob
                changes.append(
                    f"RSI thresholds tightened: oversold {old_os}→{new_os}, overbought {old_ob}→{new_ob}. "
                    f"Ranging regime win_rate={ranging_wr:.1%} vs trending_up={trending_up_wr:.1%}, "
                    f"trending_down={trending_down_wr:.1%}. Mean reversion unreliable in trending markets; "
                    "requiring stronger RSI extremes to filter trend-continuation entries."
                )

        # If overall win rate is low and enough data, reduce ATR filter to only the calmest setups
        if len(trades) >= 15 and overall_wr < 0.42:
            old_atr = float(self._config.get("max_atr_pct", 0.03))
            new_atr = max(0.015, round(old_atr - 0.005, 3))
            if new_atr != old_atr:
                self._config["max_atr_pct"] = new_atr
                changes.append(
                    f"max_atr_pct reduced {old_atr:.3f}→{new_atr:.3f}. "
                    f"Win rate {overall_wr:.1%} across {len(trades)} trades below 42% threshold. "
                    "Filtering more high-volatility names to concentrate on clean mean-reversion setups."
                )

        # If winners resolve faster than current hold_days, tighten hold period
        if avg_win_hold > 0 and avg_win_hold < self._config.get("hold_days", 5) * 0.6:
            old_hold = int(self._config.get("hold_days", 5))
            new_hold = max(2, round(avg_win_hold) + 1)
            if new_hold != old_hold:
                self._config["hold_days"] = new_hold
                changes.append(
                    f"hold_days reduced {old_hold}→{new_hold}. "
                    f"Winners resolving in avg {avg_win_hold:.1f}d vs losers {avg_loss_hold:.1f}d. "
                    "Tightening hold to lock in gains before mean-reversion exhausts."
                )

        # --- version bump & save ---
        current_v = str(self._config.get("strategy_version", "1.0"))
        try:
            major, minor = current_v.rsplit(".", 1)
            new_v = f"{major}.{int(minor) + 1}"
        except Exception:
            new_v = "1.1"

        best_regime_wr = regime_stats.get(best_regime, {}).get("win_rate", "N/A")
        worst_regime_wr = regime_stats.get(worst_regime, {}).get("win_rate", "N/A")

        description = (
            f"v{new_v} | trades={len(trades)} wr={overall_wr:.1%} "
            f"best_regime={best_regime}({best_regime_wr}) "
            f"worst={worst_regime}({worst_regime_wr}) best_ticker={best_ticker} | "
            + ("; ".join(changes) if changes else "no parameter changes — monitoring")
        )

        change_block = "\n".join(f"  - {c}" for c in changes) if changes else "  - None (monitoring phase)"
        code_lines = [
            '"""',
            f"Athena Mean Reversion Strategy — v{new_v}",
            f"Updated: {datetime.now(timezone.utc).isoformat()}",
            "",
            "Changes from previous version:",
            change_block,
            "",
            "Evidence basis:",
            f"  trades_analyzed = {len(trades)}",
            f"  overall_win_rate = {overall_wr:.3f}",
            f"  best_regime = {best_regime} (win_rate={best_regime_wr})",
            f"  worst_regime = {worst_regime} (win_rate={worst_regime_wr})",
            f"  best_ticker = {best_ticker}",
            f"  avg_win_hold_days = {avg_win_hold:.2f}",
            f"  avg_loss_hold_days = {avg_loss_hold:.2f}",
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
        logger.info("Athena updated → v%s | %s", new_v, description[:140])
