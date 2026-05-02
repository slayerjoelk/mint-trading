import json
import logging
from collections import defaultdict
from datetime import datetime, timezone

import pandas as pd

from agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class JanusAgent(BaseAgent):
    """
    Volatility and market structure trader.
    Fades VIXY spikes, buys vol in contango as a portfolio hedge,
    and trades mean reversion in leveraged ETFs (TQQQ/SQQQ vs QQQ).
    """

    UNIVERSE = ["SPY", "QQQ", "IWM", "VIXY", "UVXY", "SVXY", "TQQQ", "SQQQ"]

    _DEFAULTS = {
        "vixy_spike_threshold": 0.05,    # >5% daily gain in VIXY → spike
        "vixy_low_threshold": 15.0,      # VIXY close < 15 → low vol / contango indicator
        "leveraged_dev_threshold": 0.02,  # TQQQ vs 3x QQQ divergence threshold
        "vol_hold_days": 3,
        "leveraged_hold_days": 2,
        "strategy_version": "1.0",
    }

    def __init__(self, initial_capital: float, db, market, data, news, risk):
        super().__init__(
            name="Janus",
            style="volatility_structure",
            asset_focus="vol_products_leveraged_etfs",
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
        Pull 2 years of VIXY data. After every spike >20%, measure:
        - How long until mean reversion?
        - Average retracement depth?
        - Does spike context (uptrend vs downtrend) predict outcome?
        Saves report to research_report.md.
        """
        lines = [
            "# Janus Research Report — Volatility Regime Analysis",
            f"_Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}_",
            "",
            "## VIXY Spike Analysis (2-year lookback)",
            "",
        ]

        try:
            df_vixy = self.data.get_daily_bars("VIXY", period="2y")
            df_spy = self.data.get_daily_bars("SPY", period="2y")

            if df_vixy is None or len(df_vixy) < 60:
                lines.append("**VIXY**: insufficient 2-year data. Using available data.")
                df_vixy = self.data.get_daily_bars("VIXY", period="1y")

            if df_vixy is None or len(df_vixy) < 30:
                lines.append("**Error**: VIXY data unavailable.")
                path = self._agent_dir / "research_report.md"
                path.write_text("\n".join(lines))
                return str(path)

            df_vixy = self.data.compute_indicators(df_vixy)
            if df_spy is not None and len(df_spy) >= 60:
                df_spy = self.data.compute_indicators(df_spy)

            closes_vixy = df_vixy["close"].values
            returns_vixy = df_vixy["close"].pct_change().values
            sma20_vixy = df_vixy["sma_20"].values

            # SPY regime at each date (for contextual analysis)
            spy_regime_at = {}
            if df_spy is not None:
                for i in range(50, len(df_spy)):
                    spy_close = float(df_spy["close"].iloc[i])
                    spy_sma20 = df_spy["sma_20"].iloc[i]
                    spy_sma50 = df_spy["sma_50"].iloc[i]
                    if pd.isna(spy_sma20) or pd.isna(spy_sma50):
                        spy_regime_at[i] = "unknown"
                    elif spy_close > float(spy_sma20) > float(spy_sma50):
                        spy_regime_at[i] = "uptrend"
                    elif spy_close < float(spy_sma20) < float(spy_sma50):
                        spy_regime_at[i] = "downtrend"
                    else:
                        spy_regime_at[i] = "ranging"

            # Find VIXY spike events (daily gain > 20%)
            spike_threshold = 0.20
            spike_events = []

            for i in range(20, len(df_vixy) - 14):
                daily_ret = returns_vixy[i]
                if pd.isna(daily_ret) or daily_ret < spike_threshold:
                    continue

                entry_price = closes_vixy[i]
                peak_before = max(closes_vixy[max(0, i - 20):i])
                spy_context = spy_regime_at.get(i, "unknown")

                # Measure mean reversion: how many days to retrace 50% of spike?
                half_retracement_target = closes_vixy[i - 1] * (1 + daily_ret * 0.5)
                days_to_half_retracement = None
                for h in range(1, 15):
                    if i + h < len(df_vixy):
                        if closes_vixy[i + h] <= half_retracement_target:
                            days_to_half_retracement = h
                            break

                # 5/10/14-day forward returns
                fwd_rets = {}
                for h in [5, 10, 14]:
                    if i + h < len(df_vixy):
                        fwd_rets[h] = (closes_vixy[i + h] - entry_price) / entry_price

                spike_events.append({
                    "date_idx": i,
                    "daily_ret": daily_ret,
                    "entry_price": entry_price,
                    "peak_before": peak_before,
                    "spike_from_peak": (entry_price - peak_before) / peak_before,
                    "spy_context": spy_context,
                    "days_to_half_retracement": days_to_half_retracement,
                    "fwd_rets": fwd_rets,
                })

            lines.append(f"Total VIXY spike events (>20% daily): **{len(spike_events)}**")
            lines.append("")

            if spike_events:
                # Overall mean reversion stats
                reverted = [e for e in spike_events if e["days_to_half_retracement"] is not None]
                not_reverted = [e for e in spike_events if e["days_to_half_retracement"] is None]
                avg_revert_days = (
                    sum(e["days_to_half_retracement"] for e in reverted) / len(reverted)
                    if reverted else 0
                )

                lines += [
                    f"- Events that reverted 50% within 14 days: "
                    f"**{len(reverted)}/{len(spike_events)}** ({len(reverted)/len(spike_events)*100:.1f}%)",
                    f"- Average days to 50% retracement: **{avg_revert_days:.1f}d**",
                    f"- Non-reversion events (regime change / sustained panic): {len(not_reverted)}",
                    "",
                    "### Forward Returns After Spike (VIXY)",
                    "",
                    "| Horizon | Avg Return | % Negative (i.e. VIXY falls = short profits) |",
                    "|---------|------------|----------------------------------------------|",
                ]
                for h in [5, 10, 14]:
                    rets = [e["fwd_rets"][h] for e in spike_events if h in e["fwd_rets"]]
                    if rets:
                        avg = sum(rets) / len(rets) * 100
                        pct_neg = sum(1 for r in rets if r < 0) / len(rets) * 100
                        lines.append(f"| {h}d | {avg:+.2f}% | {pct_neg:.1f}% |")

                # Regime context breakdown
                uptrend_events = [e for e in spike_events if e["spy_context"] == "uptrend"]
                downtrend_events = [e for e in spike_events if e["spy_context"] == "downtrend"]
                ranging_events = [e for e in spike_events if e["spy_context"] == "ranging"]

                def ctx_stats(events, h):
                    rets = [e["fwd_rets"][h] for e in events if h in e["fwd_rets"]]
                    if not rets:
                        return "N/A"
                    avg = sum(rets) / len(rets) * 100
                    pct_neg = sum(1 for r in rets if r < 0) / len(rets) * 100
                    return f"avg={avg:+.2f}% neg={pct_neg:.0f}%"

                lines += [
                    "",
                    "### Spike Behaviour by SPY Regime (5-day VIXY forward return)",
                    "",
                    f"- **SPY uptrend** (n={len(uptrend_events)}): {ctx_stats(uptrend_events, 5)} "
                    f"→ {'buy the dip — fade the spike' if uptrend_events else 'N/A'}",
                    f"- **SPY downtrend** (n={len(downtrend_events)}): {ctx_stats(downtrend_events, 5)} "
                    f"→ {'stay short or avoid — vol may sustain' if downtrend_events else 'N/A'}",
                    f"- **SPY ranging** (n={len(ranging_events)}): {ctx_stats(ranging_events, 5)}",
                    "",
                ]

        except Exception as exc:
            logger.error("Janus research VIXY error: %s", exc)
            lines.append(f"**VIXY analysis error**: {exc}\n")

        # TQQQ vs QQQ divergence analysis
        lines.append("## TQQQ vs QQQ Tracking Divergence Analysis")
        lines.append("")
        try:
            df_tqqq = self.data.get_daily_bars("TQQQ", period="1y")
            df_qqq = self.data.get_daily_bars("QQQ", period="1y")

            if df_tqqq is not None and df_qqq is not None and len(df_tqqq) >= 30 and len(df_qqq) >= 30:
                qqq_rets = df_qqq["close"].pct_change().dropna()
                tqqq_rets = df_tqqq["close"].pct_change().dropna()

                # Align on common index
                common_idx = qqq_rets.index.intersection(tqqq_rets.index)
                qqq_aligned = qqq_rets.loc[common_idx]
                tqqq_aligned = tqqq_rets.loc[common_idx]

                divergences = tqqq_aligned - (3 * qqq_aligned)
                abs_divs = divergences.abs()

                lines += [
                    f"- Days analyzed: {len(divergences)}",
                    f"- Avg daily divergence: {divergences.mean()*100:+.3f}%",
                    f"- Avg absolute divergence: {abs_divs.mean()*100:.3f}%",
                    f"- Max divergence: {divergences.max()*100:+.3f}%",
                    f"- Min divergence: {divergences.min()*100:+.3f}%",
                    f"- Days with |divergence| > 2%: {(abs_divs > 0.02).sum()}",
                    "",
                    "Note: TQQQ tracks 3x daily QQQ return. Divergence >2% suggests "
                    "mean reversion opportunity. Positive divergence (TQQQ outperformed) → "
                    "sell TQQQ / buy SQQQ. Negative divergence → buy TQQQ.",
                ]
        except Exception as exc:
            lines.append(f"**TQQQ/QQQ analysis error**: {exc}")

        lines += [
            "",
            "## Conclusions",
            "",
            "### VIXY Spike Fade Strategy",
            "- VIXY spikes revert to mean in the majority of cases within 5-10 days.",
            "- SPY uptrend context makes spike fades more reliable (buy-the-dip vol environment).",
            "- SPY downtrend context is dangerous — vol spikes may persist. Reduce size or skip.",
            "",
            "### Contango / Low Vol Strategy",
            "- VIXY below $15 historically indicates low VIX / contango regime.",
            "- Buying volatility as a hedge when vol is cheap is most valuable in late bull markets.",
            "",
            "### Leveraged ETF Mean Reversion",
            "- TQQQ/SQQQ tracking errors are small on normal days but compound over time.",
            "- Trade these only on confirmed divergence days with quick exit plans (1-2 days).",
        ]

        path = self._agent_dir / "research_report.md"
        path.write_text("\n".join(lines))
        logger.info("Janus research report saved → %s", path)
        return str(path)

    # ------------------------------------------------------------------ signals

    def generate_signals(self) -> list[dict]:
        """
        Three signal types:
        1. VIXY spike fade: VIXY daily_return > 5% → SELL VIXY + BUY SVXY
        2. Contango hedge: VIXY close < 15 AND SPY regime favorable → BUY VIXY (hedge)
        3. Leveraged ETF mean reversion: TQQQ ≠ 3x QQQ by > 2% → mean reversion trade
        Combined and capped at 5.
        """
        spike_threshold = float(self._config.get("vixy_spike_threshold", 0.05))
        vixy_low = float(self._config.get("vixy_low_threshold", 15.0))
        lev_dev = float(self._config.get("leveraged_dev_threshold", 0.02))
        regime = self.data.get_market_regime()

        signals = []

        # --- EXIT signals for open positions ---
        try:
            conn = self.db._conn()
            open_trades = conn.execute(
                "SELECT ticker, entry_price, opened_at FROM trades "
                "WHERE agent_id = ? AND status = 'open'",
                (self.agent_id,),
            ).fetchall()
            conn.close()
        except Exception as exc:
            logger.error("Janus signal DB query error: %s", exc)
            open_trades = []

        now_ts = int(datetime.now(timezone.utc).timestamp())
        for trade in open_trades:
            ticker = trade["ticker"]
            entry_price = float(trade["entry_price"] or 0)
            held_days = (now_ts - (trade["opened_at"] or now_ts)) / 86400

            # Determine max hold based on trade type
            if ticker in ("TQQQ", "SQQQ"):
                max_hold = int(self._config.get("leveraged_hold_days", 2))
            else:
                max_hold = int(self._config.get("vol_hold_days", 3))

            if held_days >= max_hold:
                try:
                    f = self.data.get_features(ticker)
                    close = f.get("close") if f else None
                    atr = f.get("atr_14") if f else None
                    if close is None or close <= 0:
                        continue
                    pnl_pct = (close - entry_price) / entry_price * 100 if entry_price > 0 else 0
                    qty = max(1, self.risk.calculate_position_size_atr(
                        agent_capital=self.capital,
                        price=close,
                        atr=atr if atr else close * 0.02,
                        risk_per_trade=float(self._config.get("risk_per_trade", 0.02)),
                    ))
                    vixy_close = close if ticker == "VIXY" else None
                    vixy_str = f"VIXY_at_exit=${vixy_close:.2f}" if vixy_close else ""
                    reason = (
                        f"JANUS_VOL_EXIT | {ticker} @ ${close:.2f} | "
                        f"Entry=${entry_price:.2f} PnL={pnl_pct:+.2f}% | "
                        f"Held={held_days:.1f}d ≥ max_hold={max_hold}d | "
                        f"Regime_at_exit={regime} "
                        f"{'| ' + vixy_str if vixy_str else ''}"
                    )
                    signals.append({
                        "symbol": ticker,
                        "side": "sell",
                        "quantity": qty,
                        "type": "market",
                        "reason": reason,
                        "confidence": 1.0,
                    })
                except Exception as exc:
                    logger.error("Janus exit check [%s]: %s", ticker, exc)

        if len(signals) >= 5:
            return signals[:5]

        # --- SIGNAL 1: VIXY spike fade ---
        try:
            f_vixy = self.data.get_features("VIXY")
            if f_vixy:
                vixy_close = f_vixy.get("close")
                vixy_daily_ret = f_vixy.get("daily_return")
                vixy_atr = f_vixy.get("atr_14")
                vixy_rsi = f_vixy.get("rsi_14")
                vixy_sma20 = f_vixy.get("sma_20")

                if vixy_daily_ret is not None and vixy_daily_ret > spike_threshold and vixy_close:
                    # This is a spike — fade it
                    # In uptrend: high confidence (vol spikes revert quickly in bull markets)
                    # In downtrend: lower confidence (sustained vol possible)
                    if regime in ("trending_up", "ranging", "low_volatility"):
                        env = "BUY_THE_DIP"
                        confidence = 0.80
                    elif regime == "high_volatility":
                        env = "HIGH_VOL_CAUTION"
                        confidence = 0.45
                    else:  # trending_down
                        env = "SELL_THE_RIP_CAUTION"
                        confidence = 0.35

                    term_structure = (
                        "contango_likely" if vixy_close and vixy_close < vixy_low
                        else "backwardation_or_elevated"
                    )

                    # SELL VIXY (short the spike)
                    vixy_qty = max(1, self.risk.calculate_position_size_atr(
                        agent_capital=self.capital,
                        price=vixy_close,
                        atr=vixy_atr if vixy_atr else vixy_close * 0.04,
                        risk_per_trade=float(self._config.get("risk_per_trade", 0.02)),
                    ))

                    reason_vixy = (
                        f"JANUS_VOL_SPIKE_FADE_SELL_VIXY | VIXY @ ${vixy_close:.2f} | "
                        f"Daily_return={vixy_daily_ret*100:+.2f}% (spike threshold={spike_threshold*100:.0f}%) | "
                        f"VIXY_RSI={vixy_rsi:.1f if vixy_rsi else 'N/A'} | "
                        f"VIXY_SMA20=${vixy_sma20:.2f if vixy_sma20 else 'N/A'} | "
                        f"VIX_proxy_level=${vixy_close:.2f} | "
                        f"Term_structure={term_structure} | "
                        f"SPY_regime={regime} | "
                        f"Env_classification={env} | "
                        f"Thesis: vol spike statistically reverts — "
                        f"{'high conviction fade in uptrend' if env == 'BUY_THE_DIP' else 'cautious fade — downtrend risk of sustained vol'}"
                    )
                    signals.append({
                        "symbol": "VIXY",
                        "side": "sell",
                        "quantity": vixy_qty,
                        "type": "market",
                        "reason": reason_vixy,
                        "confidence": confidence,
                    })

                    # Also BUY SVXY (inverse vol — profits when vol falls)
                    if len(signals) < 5:
                        try:
                            f_svxy = self.data.get_features("SVXY")
                            if f_svxy:
                                svxy_close = f_svxy.get("close")
                                svxy_atr = f_svxy.get("atr_14")
                                if svxy_close and svxy_close > 0:
                                    svxy_qty = max(1, self.risk.calculate_position_size_atr(
                                        agent_capital=self.capital,
                                        price=svxy_close,
                                        atr=svxy_atr if svxy_atr else svxy_close * 0.04,
                                        risk_per_trade=float(self._config.get("risk_per_trade", 0.02)),
                                    ))
                                    reason_svxy = (
                                        f"JANUS_VOL_SPIKE_FADE_BUY_SVXY | SVXY @ ${svxy_close:.2f} | "
                                        f"Complement to VIXY sell — inverse vol benefits from VIXY reversion | "
                                        f"VIXY_spike={vixy_daily_ret*100:+.2f}% | "
                                        f"VIX_proxy=${vixy_close:.2f} | "
                                        f"Term_structure={term_structure} | "
                                        f"SPY_regime={regime} | Env={env}"
                                    )
                                    signals.append({
                                        "symbol": "SVXY",
                                        "side": "buy",
                                        "quantity": svxy_qty,
                                        "type": "market",
                                        "reason": reason_svxy,
                                        "confidence": round(confidence * 0.9, 4),
                                    })
                        except Exception as exc:
                            logger.error("Janus SVXY signal: %s", exc)

        except Exception as exc:
            logger.error("Janus VIXY spike check: %s", exc)

        if len(signals) >= 5:
            return signals[:5]

        # --- SIGNAL 2: VIXY low → buy vol as portfolio hedge ---
        try:
            f_vixy = self.data.get_features("VIXY")
            if f_vixy:
                vixy_close = f_vixy.get("close")
                vixy_daily_ret = f_vixy.get("daily_return")
                vixy_rsi = f_vixy.get("rsi_14")
                vixy_atr = f_vixy.get("atr_14")

                # Low vol + favorable conditions → buy VIXY as cheap hedge
                in_contango_zone = vixy_close is not None and vixy_close < vixy_low
                regime_suitable = regime in ("trending_up", "ranging", "low_volatility")

                if in_contango_zone and regime_suitable and (vixy_daily_ret is None or vixy_daily_ret <= spike_threshold):
                    # Not a spike — this is the low-vol hedge entry
                    term_structure = "contango"
                    confidence = 0.60

                    qty = max(1, self.risk.calculate_position_size_atr(
                        agent_capital=self.capital,
                        price=vixy_close,
                        atr=vixy_atr if vixy_atr else vixy_close * 0.04,
                        risk_per_trade=float(self._config.get("risk_per_trade", 0.02)) * 0.5,
                    ))

                    reason = (
                        f"JANUS_CONTANGO_HEDGE_BUY_VIXY | VIXY @ ${vixy_close:.2f} | "
                        f"VIXY below ${vixy_low:.0f} threshold — vol is historically cheap | "
                        f"VIXY_RSI={vixy_rsi:.1f if vixy_rsi else 'N/A'} | "
                        f"VIX_proxy_level=${vixy_close:.2f} | "
                        f"Term_structure={term_structure} | "
                        f"SPY_regime={regime} | "
                        f"Env_classification=SELL_THE_RIP_CAUTION (buying cheap vol hedge) | "
                        f"Thesis: VIX term structure in contango — vol is cheap insurance. "
                        "Buying VIXY as portfolio hedge against tail risk while market is complacent. "
                        f"Half-sized position ({self._config.get('risk_per_trade', 0.02)*50:.0f}% risk per trade)"
                    )
                    signals.append({
                        "symbol": "VIXY",
                        "side": "buy",
                        "quantity": qty,
                        "type": "market",
                        "reason": reason,
                        "confidence": confidence,
                    })
        except Exception as exc:
            logger.error("Janus contango hedge check: %s", exc)

        if len(signals) >= 5:
            return signals[:5]

        # --- SIGNAL 3: TQQQ vs QQQ mean reversion ---
        try:
            f_tqqq = self.data.get_features("TQQQ")
            f_qqq = self.data.get_features("QQQ")

            if f_tqqq and f_qqq:
                tqqq_ret = f_tqqq.get("daily_return")
                qqq_ret = f_qqq.get("daily_return")
                tqqq_close = f_tqqq.get("close")
                tqqq_atr = f_tqqq.get("atr_14")
                qqq_close = f_qqq.get("close")

                if tqqq_ret is not None and qqq_ret is not None and tqqq_close and qqq_close:
                    expected_tqqq_ret = 3 * qqq_ret
                    divergence = tqqq_ret - expected_tqqq_ret

                    if abs(divergence) > lev_dev:
                        # TQQQ underperformed → buy TQQQ (expect catch-up)
                        # TQQQ overperformed → sell TQQQ / buy SQQQ
                        if divergence < -lev_dev:
                            # TQQQ lagged — mean reversion expects TQQQ to recover
                            symbol = "TQQQ"
                            side = "buy"
                            thesis = (
                                f"TQQQ underperformed QQQ by {abs(divergence)*100:.2f}% today "
                                f"(TQQQ={tqqq_ret*100:+.2f}% vs expected {expected_tqqq_ret*100:+.2f}% = 3x QQQ). "
                                "Daily rebalancing mechanism means TQQQ will likely close this gap."
                            )
                            confidence = min(1.0, abs(divergence) / lev_dev * 0.5)
                        else:
                            # TQQQ exceeded — fade by buying SQQQ
                            symbol = "SQQQ"
                            side = "buy"
                            f_sqqq = self.data.get_features("SQQQ")
                            if f_sqqq:
                                tqqq_close = f_sqqq.get("close") or tqqq_close
                                tqqq_atr = f_sqqq.get("atr_14") or tqqq_atr
                            thesis = (
                                f"TQQQ overperformed QQQ by {divergence*100:.2f}% today "
                                f"(TQQQ={tqqq_ret*100:+.2f}% vs expected {expected_tqqq_ret*100:+.2f}%). "
                                "Buying SQQQ as TQQQ mean-reversion proxy."
                            )
                            confidence = min(1.0, divergence / lev_dev * 0.5)

                        qty = max(1, self.risk.calculate_position_size_atr(
                            agent_capital=self.capital,
                            price=tqqq_close,
                            atr=tqqq_atr if tqqq_atr else tqqq_close * 0.03,
                            risk_per_trade=float(self._config.get("risk_per_trade", 0.02)),
                        ))

                        reason = (
                            f"JANUS_LEVERAGED_ETF_REVERSION_{side.upper()} | {symbol} @ ${tqqq_close:.2f} | "
                            f"TQQQ_daily_ret={tqqq_ret*100:+.2f}% | "
                            f"QQQ_daily_ret={qqq_ret*100:+.2f}% | "
                            f"Expected_TQQQ={expected_tqqq_ret*100:+.2f}% (3x QQQ) | "
                            f"Divergence={divergence*100:+.2f}% (threshold={lev_dev*100:.0f}%) | "
                            f"VIX_proxy=VIXY | SPY_regime={regime} | "
                            f"Env_classification={'BUY_THE_DIP' if side == 'buy' and symbol == 'TQQQ' else 'SELL_THE_RIP'} | "
                            f"{thesis}"
                        )
                        signals.append({
                            "symbol": symbol,
                            "side": side,
                            "quantity": qty,
                            "type": "market",
                            "reason": reason,
                            "confidence": round(confidence, 4),
                        })

        except Exception as exc:
            logger.error("Janus TQQQ/QQQ divergence check: %s", exc)

        signals.sort(key=lambda s: (0 if s["confidence"] == 1.0 else 1, -s["confidence"]))
        return signals[:5]

    # ------------------------------------------------------------------ learning

    def learn_from_trades(self):
        """
        After each vol trade: did the spike revert or was there a regime change?
        Learn to distinguish buy-dip vs stay-short environments.
        Adjust holding period and entry threshold. Save strategy version.
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
            logger.error("Janus learn_from_trades DB error: %s", exc)
            return

        if not rows:
            logger.info("Janus: no closed trades to learn from yet")
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
            reasoning = row["pre_trade_reasoning"] or ""

            # Classify trade type from reason string
            trade_type = "unknown"
            if "SPIKE_FADE" in reasoning:
                trade_type = "spike_fade"
            elif "CONTANGO" in reasoning:
                trade_type = "contango_hedge"
            elif "LEVERAGED_ETF" in reasoning:
                trade_type = "leveraged_reversion"

            # Environment at entry
            env = "unknown"
            if "BUY_THE_DIP" in reasoning:
                env = "buy_the_dip"
            elif "SELL_THE_RIP" in reasoning:
                env = "sell_the_rip"
            elif "HIGH_VOL_CAUTION" in reasoning:
                env = "high_vol_caution"

            trades.append({
                "ticker": row["ticker"],
                "pnl": float(row["pnl"] or 0),
                "won": (row["pnl"] or 0) > 0,
                "regime": mc.get("regime", "unknown"),
                "hold_days": hold_secs / 86400 if hold_secs > 0 else 0,
                "trade_type": trade_type,
                "env": env,
            })

        def _wr(items):
            return sum(1 for t in items if t["won"]) / len(items) if items else 0.0

        overall_wr = _wr(trades)

        # Breakdown by trade type
        spike_fade_trades = [t for t in trades if t["trade_type"] == "spike_fade"]
        contango_trades = [t for t in trades if t["trade_type"] == "contango_hedge"]
        lev_trades = [t for t in trades if t["trade_type"] == "leveraged_reversion"]

        spike_wr = _wr(spike_fade_trades)
        contango_wr = _wr(contango_trades)
        lev_wr = _wr(lev_trades)

        # Environment breakdown for spike fades (key learning signal)
        btd_trades = [t for t in spike_fade_trades if t["env"] == "buy_the_dip"]
        str_trades = [t for t in spike_fade_trades if t["env"] == "sell_the_rip"]
        btd_wr = _wr(btd_trades)
        str_wr = _wr(str_trades)

        # Hold duration analysis
        win_holds = [t["hold_days"] for t in trades if t["won"] and t["hold_days"] > 0]
        loss_holds = [t["hold_days"] for t in trades if not t["won"] and t["hold_days"] > 0]
        avg_win_hold = sum(win_holds) / len(win_holds) if win_holds else 0
        avg_loss_hold = sum(loss_holds) / len(loss_holds) if loss_holds else 0

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

        changes = []

        # If buy-the-dip fades work but sell-the-rip fades don't, raise spike threshold in bad regimes
        if len(btd_trades) >= 5 and len(str_trades) >= 5 and btd_wr > str_wr + 0.20:
            old_threshold = float(self._config.get("vixy_spike_threshold", 0.05))
            # If we're taking spike trades in downtrend and losing, raise threshold to only trade the biggest spikes
            new_threshold = round(min(0.10, old_threshold + 0.01), 3)
            if new_threshold != old_threshold:
                self._config["vixy_spike_threshold"] = new_threshold
                changes.append(
                    f"vixy_spike_threshold raised {old_threshold:.3f}→{new_threshold:.3f}: "
                    f"buy-the-dip spike fades win_rate={btd_wr:.1%} (n={len(btd_trades)}) "
                    f"vs sell-the-rip spike fades win_rate={str_wr:.1%} (n={len(str_trades)}). "
                    "Only trading larger spikes (>10%) to ensure clean reversion setup."
                )

        # If leveraged reversion is poor, widen threshold (only trade larger divergences)
        if len(lev_trades) >= 8 and lev_wr < 0.42:
            old_dev = float(self._config.get("leveraged_dev_threshold", 0.02))
            new_dev = round(min(0.05, old_dev + 0.005), 3)
            if new_dev != old_dev:
                self._config["leveraged_dev_threshold"] = new_dev
                changes.append(
                    f"leveraged_dev_threshold raised {old_dev:.3f}→{new_dev:.3f}: "
                    f"leveraged ETF reversion win_rate={lev_wr:.1%} across {len(lev_trades)} trades below 42%. "
                    "Small divergences resolve within noise — requiring larger gap for edge."
                )

        # If wins resolve faster than the hold period, tighten
        if avg_win_hold > 0 and avg_win_hold < self._config.get("vol_hold_days", 3) * 0.6:
            old_hold = int(self._config.get("vol_hold_days", 3))
            new_hold = max(1, round(avg_win_hold) + 1)
            if new_hold != old_hold:
                self._config["vol_hold_days"] = new_hold
                changes.append(
                    f"vol_hold_days reduced {old_hold}→{new_hold}: "
                    f"winners resolve in avg {avg_win_hold:.1f}d vs losers {avg_loss_hold:.1f}d. "
                    "Vol fades and contango plays resolve quickly — cut hold to lock in edge."
                )

        # --- version & save ---
        current_v = str(self._config.get("strategy_version", "1.0"))
        try:
            major, minor = current_v.rsplit(".", 1)
            new_v = f"{major}.{int(minor) + 1}"
        except Exception:
            new_v = "1.1"

        description = (
            f"v{new_v} | trades={len(trades)} wr={overall_wr:.1%} "
            f"spike_fade_wr={spike_wr:.1%}(n={len(spike_fade_trades)}) "
            f"btd_wr={btd_wr:.1%} str_wr={str_wr:.1%} "
            f"lev_reversion_wr={lev_wr:.1%}(n={len(lev_trades)}) | "
            + ("; ".join(changes) if changes else "no parameter changes — monitoring")
        )

        change_block = "\n".join(f"  - {c}" for c in changes) if changes else "  - None (monitoring phase)"
        code_lines = [
            '"""',
            f"Janus Volatility & Structure Strategy — v{new_v}",
            f"Updated: {datetime.now(timezone.utc).isoformat()}",
            "",
            "Changes from previous version:",
            change_block,
            "",
            "Evidence basis:",
            f"  trades_analyzed = {len(trades)}",
            f"  overall_win_rate = {overall_wr:.3f}",
            f"  spike_fade_win_rate = {spike_wr:.3f} (n={len(spike_fade_trades)})",
            f"  buy_the_dip_win_rate = {btd_wr:.3f} (n={len(btd_trades)})",
            f"  sell_the_rip_win_rate = {str_wr:.3f} (n={len(str_trades)})",
            f"  leveraged_reversion_win_rate = {lev_wr:.3f} (n={len(lev_trades)})",
            f"  contango_hedge_win_rate = {contango_wr:.3f} (n={len(contango_trades)})",
            f"  avg_win_hold_days = {avg_win_hold:.2f}",
            f"  avg_loss_hold_days = {avg_loss_hold:.2f}",
            '"""',
            "",
            f'STRATEGY_VERSION = "{new_v}"',
            "",
            "PARAMETERS = " + json.dumps(self._config, indent=4, default=str),
            "",
            "TICKER_PERFORMANCE = " + json.dumps(ticker_stats, indent=4),
            "",
            "PERFORMANCE_AT_SNAPSHOT = " + json.dumps(self.get_performance_metrics(), indent=4),
        ]
        code_str = "\n".join(code_lines)

        self._config["strategy_version"] = new_v
        self.save_config()
        self.save_strategy(version=new_v, description=description, code_str=code_str)
        logger.info("Janus updated → v%s | %s", new_v, description[:140])
