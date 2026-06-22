#!/usr/bin/env python3
"""Mercury — Crypto Momentum & Structure Agent for Mint Trading Company."""
import json
import logging
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np

from agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)

_UNIVERSE = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT",
             "ADA/USDT", "DOGE/USDT", "AVAX/USDT", "MATIC/USDT", "LINK/USDT"]

_DEFAULTS = {
    "trend_sma_short": 20, "trend_sma_long": 50,
    "mr_rsi_oversold": 30, "mr_rsi_overbought": 70,
    "mr_bb_period": 20, "mr_bb_std": 2.0,
    "btc_dom_threshold": 0.05,
    "max_hold_days": 7,
    "strategy_version": "1.0",
}


class MercuryAgent(BaseAgent):
    """Multi-strategy crypto agent. Trend following + mean reversion + BTC dominance rotation."""

    UNIVERSE = _UNIVERSE

    def __init__(self, initial_capital, db, market, data, news, risk, crypto_market=None, shared_knowledge=None):
        # accept extra kwargs so croesus can pass crypto_market + shared_knowledge
        super().__init__(
            name="Mercury", style="crypto_momentum_structure",
            asset_focus="crypto_top_10",
            initial_capital=initial_capital,
            db=db, market=market, data=data, news=news, risk=risk,
        )
        self.crypto = crypto_market
        self.sk = shared_knowledge
        for k, v in _DEFAULTS.items():
            if k not in self._config:
                self._config[k] = v
        self.save_config()

    def _ticker_to_symbol(self, pair: str) -> str:
        """BTC/USDT → BTC/USD for Alpaca crypto trading. Alpaca accepts BTC/USD or BTCUSD, NOT BTC-USD."""
        base = pair.split("/")[0].upper()
        return f"{base}/USD"

    # ──────────────────────────────────────────────────────────────── research

    def research(self) -> str:
        pairs = self.UNIVERSE
        lines = [
            "# Mercury Research Report — Crypto Multi-Strategy",
            f"_Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}_",
            "",
            "## Data Sources",
            f"- ccxt OHLCV (Binance) for {len(pairs)} pairs, 1-year lookback.",
            "",
        ]
        if not self.crypto:
            lines.append("**No crypto market interface available.**")
            path = self._agent_dir / "research_report.md"
            path.write_text("\n".join(lines))
            return str(path)

        # Trend strategy analysis per pair
        lines.append("## Trend Strategy (BTC-led, SMA crossover)")
        lines.append("")
        for pair in pairs:
            ohlcv = self.crypto.get_ohlcv(pair, "1d", 365)
            if len(ohlcv) < 60:
                lines.append(f"- **{pair}**: insufficient data")
                continue
            closes = [c[4] for c in ohlcv]
            sma20 = self._sma(closes, 20)
            sma50 = self._sma(closes, 50)
            if len(sma20) < 30:
                continue
            wins = 0
            total = 0
            for i in range(50, len(closes) - 7):
                if sma20[i] and sma50[i] and sma20[i] > sma50[i]:
                    ret = (closes[i + 7] - closes[i]) / closes[i]
                    total += 1
                    if ret > 0:
                        wins += 1
            wr = wins / total * 100 if total > 0 else 0
            lines.append(f"- **{pair}**: {total} trend signals, win rate: {wr:.1f}%")

        # Mean reversion analysis
        lines.append("")
        lines.append("## Mean Reversion Strategy (RSI + BB on altcoins)")
        alt_pairs = [p for p in pairs if p not in ("BTC/USDT", "ETH/USDT")]
        for pair in alt_pairs:
            ohlcv = self.crypto.get_ohlcv(pair, "1d", 365)
            if len(ohlcv) < 40:
                continue
            closes = np.array([c[4] for c in ohlcv])
            rsi = self._compute_rsi(closes, 14)
            bb_lower, bb_mid = self._compute_bb(closes, 20, 2.0)
            wins = 0
            total = 0
            for i in range(30, len(closes) - 5):
                if rsi[i] is not None and bb_lower[i] is not None:
                    if rsi[i] < 30 and closes[i] < bb_lower[i]:
                        ret = (closes[i + 5] - closes[i]) / closes[i]
                        total += 1
                        if ret > 0:
                            wins += 1
            wr = wins / total * 100 if total > 0 else 0
            lines.append(f"- **{pair}**: {total} MR signals, 5d win rate: {wr:.1f}%")

        path = self._agent_dir / "research_report.md"
        path.write_text("\n".join(lines))
        logger.info("Mercury research report saved → %s", path)
        return str(path)

    # ──────────────────────────────────────────────────────────────── signals

    def generate_signals(self) -> list:
        if not self.crypto:
            return []
        signals = []
        pairs = self.UNIVERSE
        
        # Get the latest bar open time for deduplication (using crypto 1d bars)
        bar_open_time = None
        ohlcv = self.crypto.get_ohlcv("BTC/USDT", "1d", 2)
        if ohlcv and len(ohlcv) > 0:
            # ccxt OHLCV: [timestamp, open, high, low, close, volume]
            # timestamp is in milliseconds
            bar_open_time = ohlcv[-1][0] // 1000  # convert to seconds

        btc_ticker = self.crypto.get_ticker("BTC/USDT")
        btc_price = btc_ticker.get("last", 0)
        eth_ticker = self.crypto.get_ticker("ETH/USDT")
        eth_price = eth_ticker.get("last", 0)
        btc_dominance = btc_price / (btc_price + eth_price) if btc_price and eth_price else 0.5

        for pair in pairs:
            ohlcv = self.crypto.get_ohlcv(pair, "1d", 60)
            if len(ohlcv) < 30:
                continue
            closes = np.array([c[4] for c in ohlcv])
            current = closes[-1]
            if current <= 0:
                continue

            sma20 = float(np.mean(closes[-20:]))
            sma50 = float(np.mean(closes[-min(50, len(closes)):]))
            rsi = self._compute_rsi(closes, 14)[-1]
            bb_lower_list, bb_mid_list = self._compute_bb(closes, 20, 2.0)
            bb_lower = bb_lower_list[-1] if bb_lower_list else None
            bb_mid = bb_mid_list[-1] if bb_mid_list else None

            # Trend strategy: above SMAs → long on pullbacks
            if current > sma20 > sma50 and rsi is not None:
                pullback = (current - sma20) / sma20 if sma20 > 0 else 0
                if -0.03 < pullback < 0.01:
                    max_usd = min(self.capital * 0.10, 50)
                    qty = round(max_usd / current, 6)
                    signals.append({
                        "symbol": self._ticker_to_symbol(pair), "side": "buy",
                        "quantity": qty, "type": "market",
                        "reason": f"MERCURY_TREND | {pair} @ ${current:.2f} | Pullback to SMA20 (deviation={pullback*100:+.1f}%) | BTC_dominance={btc_dominance:.2f}",
                        "confidence": 0.65,
                        "bar_open_time": bar_open_time,
                    })

            # Mean reversion: RSI oversold + below BB lower
            if rsi is not None and bb_lower is not None and rsi < 30 and current < bb_lower:
                max_usd = min(self.capital * 0.08, 40)
                qty = round(max_usd / current, 6)
                deviation = (bb_lower - current) / bb_lower * 100
                signals.append({
                    "symbol": self._ticker_to_symbol(pair), "side": "buy",
                    "quantity": qty, "type": "market",
                    "reason": f"MERCURY_MR | {pair} @ ${current:.2f} | RSI={rsi:.1f} below BB_lower=${bb_lower:.2f} ({deviation:.1f}% below) | Target BB_mid=${bb_mid:.2f}",
                    "confidence": 0.75,
                    "bar_open_time": bar_open_time,
                })

            # BTC dominance rotation
            if pair == "BTC/USDT" and btc_dominance > 0.55:
                max_usd = min(self.capital * 0.10, 50)
                qty = round(max_usd / current, 6)
                signals.append({
                    "symbol": self._ticker_to_symbol(pair), "side": "buy",
                    "quantity": qty, "type": "market",
                    "reason": f"MERCURY_BTC_ROTATION | BTC dominance {btc_dominance:.2f} > 0.55 — rotating to BTC over alts",
                    "confidence": 0.60,
                    "bar_open_time": bar_open_time,
                })

        sorted_signals = sorted(signals, key=lambda s: s["confidence"], reverse=True)[:5]
        # Filter out signals from bars we've already acted on
        return self._filter_deduplicated_signals(sorted_signals)

    # ──────────────────────────────────────────────────────────────── learning

    def learn_from_trades(self):
        try:
            conn = self.db._conn()
            rows = conn.execute(
                "SELECT t.ticker, t.pnl, j.market_conditions FROM trades t "
                "LEFT JOIN trade_journal j ON j.trade_id = t.id "
                "WHERE t.agent_id = ? AND t.status = 'closed' "
                "ORDER BY t.closed_at DESC LIMIT 60",
                (self.agent_id,),
            ).fetchall()
            conn.close()
        except Exception:
            return

        if not rows:
            return

        pnls = [float(r["pnl"] or 0) for r in rows]
        wins = [p for p in pnls if p > 0]
        wr = len(wins) / len(pnls) if pnls else 0

        changes = []
        if len(pnls) >= 10 and wr < 0.42:
            old_hold = int(self._config.get("max_hold_days", 7))
            new_hold = max(3, old_hold - 2)
            self._config["max_hold_days"] = new_hold
            changes.append(f"max_hold_days {old_hold}→{new_hold}: overall WR={wr:.1%}")

        try:
            major, minor = str(self._config.get("strategy_version", "1.0")).rsplit(".", 1)
            new_v = f"{major}.{int(minor) + 1}"
        except Exception:
            new_v = "1.1"

        desc = f"v{new_v} | trades={len(pnls)} wr={wr:.1%} " + ("; ".join(changes) if changes else "monitoring")
        self._config["strategy_version"] = new_v
        self.save_config()
        self.save_strategy(new_v, desc, f"# Mercury v{new_v}\n{desc}\n")

        # Contribute to shared knowledge
        if self.sk and len(pnls) >= 5:
            btc = self.crypto.get_ticker("BTC/USDT") if self.crypto else {}
            current_btc = btc.get("last", 0)
            btc_trend = "uptrend" if current_btc else "unknown"
            self.sk.add_insight(
                self.agent_id, self.name, "crypto_regime",
                f"In {btc_trend} BTC regime, crypto WR={wr:.1%} across {len(pnls)} trades",
                {"win_rate": wr, "btc_price": current_btc}, 0.7, ticker="BTC/USDT",
            )
        logger.info("Mercury learned → v%s | wr=%.1f%% | %d trades", new_v, wr * 100, len(pnls))

    # ─────────────────────────────────────────────────────── helpers

    @staticmethod
    def _sma(data, period):
        out = [None] * len(data)
        for i in range(period - 1, len(data)):
            out[i] = float(np.mean(data[i - period + 1:i + 1]))
        return out

    @staticmethod
    def _compute_rsi(closes, period=14):
        if len(closes) < period + 1:
            return [None] * len(closes)
        deltas = np.diff(closes)
        rsi = [None] * (period + 1)
        gains = np.maximum(deltas[:period], 0)
        losses = np.maximum(-deltas[:period], 0)
        avg_gain = float(np.mean(gains))
        avg_loss = float(np.mean(losses))
        if avg_loss == 0:
            rsi_val = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi_val = 100.0 - (100.0 / (1.0 + rs))
        rsi.append(rsi_val)
        for i in range(period + 1, len(closes)):
            delta = deltas[i - 1]
            gain = delta if delta > 0 else 0
            loss = -delta if delta < 0 else 0
            avg_gain = (avg_gain * (period - 1) + gain) / period
            avg_loss = (avg_loss * (period - 1) + loss) / period
            if avg_loss == 0:
                rsi_val = 100.0
            else:
                rs = avg_gain / avg_loss
                rsi_val = 100.0 - (100.0 / (1.0 + rs))
            rsi.append(rsi_val)
        return rsi

    @staticmethod
    def _compute_bb(closes, period=20, std_mult=2.0):
        bb_lower = [None] * len(closes)
        bb_mid = [None] * len(closes)
        for i in range(period - 1, len(closes)):
            window = closes[i - period + 1:i + 1]
            mid = float(np.mean(window))
            std = float(np.std(window))
            bb_mid[i] = mid
            bb_lower[i] = mid - std_mult * std
        return bb_lower, bb_mid
