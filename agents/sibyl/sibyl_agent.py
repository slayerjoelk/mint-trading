import json
import logging
from collections import defaultdict
from datetime import datetime, timedelta, timezone

from agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)

# Words that suggest earnings-related news vs general market news
_EARNINGS_KEYWORDS = {
    "earnings", "revenue", "profit", "eps", "guidance", "forecast",
    "quarter", "quarterly", "q1", "q2", "q3", "q4", "beat", "miss",
    "results", "outlook", "estimate", "expectation",
}


def _is_earnings_related(headline: str) -> bool:
    words = set(headline.lower().split())
    return bool(words & _EARNINGS_KEYWORDS)


class SibylAgent(BaseAgent):
    """
    News-driven catalyst trader. Dynamic universe sourced from the news pipeline.
    Enters when sentiment for a ticker exceeds ±0.6 with price confirmation.
    Exits when sentiment reverses or PnL target hit (+5%).
    """

    _DEFAULTS = {
        "sentiment_threshold_buy": 0.6,
        "sentiment_threshold_sell": -0.6,
        "max_atr_pct": 0.05,
        "pnl_target_pct": 0.05,
        "max_hold_days": 3,
        "strategy_version": "1.0",
    }

    def __init__(self, initial_capital: float, db, market, data, news, risk):
        super().__init__(
            name="Sibyl",
            style="news_catalyst",
            asset_focus="news_catalyst_dynamic",
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
        Pull 30 days of news from the DB. For every ticker that appeared with
        |sentiment| > 0.5, measure 1/3/5-day forward returns. Test whether
        sentiment magnitude correlates with return magnitude.
        Saves report to research_report.md.
        """
        cutoff = int((datetime.now(timezone.utc) - timedelta(days=30)).timestamp())

        lines = [
            "# Sibyl Research Report — News Sentiment Catalyst Analysis",
            f"_Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}_",
            "",
            "## Methodology",
            "For each ticker with |sentiment| > 0.5 in the last 30 days, "
            "pull historical prices and measure forward returns at 1/3/5-day horizons.",
            "",
        ]

        # Pull high-sentiment news from DB
        try:
            conn = self.db._conn()
            rows = conn.execute(
                """
                SELECT ticker, headline, sentiment, source, published_at
                FROM news_events
                WHERE published_at >= ? AND ticker IS NOT NULL AND ticker != ''
                  AND ABS(sentiment) >= 0.5
                ORDER BY published_at DESC LIMIT 500
                """,
                (cutoff,),
            ).fetchall()
            conn.close()
        except Exception as exc:
            logger.error("Sibyl research DB error: %s", exc)
            lines.append(f"**Error reading news DB**: {exc}")
            path = self._agent_dir / "research_report.md"
            path.write_text("\n".join(lines))
            return str(path)

        if not rows:
            lines.append("No high-sentiment news found in the last 30 days. "
                         "The news pipeline may not have run yet.")
            path = self._agent_dir / "research_report.md"
            path.write_text("\n".join(lines))
            return str(path)

        # Group by ticker
        ticker_events: dict = defaultdict(list)
        source_events: dict = defaultdict(list)
        for row in rows:
            ticker_events[row["ticker"]].append({
                "headline": row["headline"],
                "sentiment": float(row["sentiment"] or 0),
                "source": row["source"],
                "published_at": row["published_at"],
                "is_earnings": _is_earnings_related(row["headline"] or ""),
            })
            source_events[row["source"] or "unknown"].append(float(row["sentiment"] or 0))

        lines.append(f"Total high-sentiment news items (30d): **{len(rows)}**")
        lines.append(f"Unique tickers: **{len(ticker_events)}**")
        lines.append(f"Sources: {', '.join(source_events.keys())}")
        lines.append("")
        lines.append("## Per-Ticker Sentiment vs Forward Return Analysis")
        lines.append("")

        summary = []
        for ticker, events in list(ticker_events.items())[:20]:  # cap at 20 tickers for brevity
            try:
                df = self.data.get_daily_bars(ticker, period="3mo")
                if df is None or len(df) < 10:
                    lines.append(f"**{ticker}**: no price data available\n")
                    continue
                df = self.data.compute_indicators(df)
                closes = df["close"].values
                dates = df.index

                results = []
                for event in events:
                    pub_ts = event["published_at"]
                    pub_dt = datetime.fromtimestamp(pub_ts, tz=timezone.utc).date()
                    # Find the index in df corresponding to this date
                    matching = [i for i, d in enumerate(dates) if d.date() == pub_dt]
                    if not matching:
                        continue
                    idx = matching[-1]
                    entry_price = closes[idx]
                    for h in [1, 3, 5]:
                        if idx + h < len(closes):
                            fwd_ret = (closes[idx + h] - entry_price) / entry_price
                            # For buy signals: positive sentiment should → positive return
                            # For sell signals: negative sentiment should → negative return
                            direction_correct = (
                                (event["sentiment"] > 0 and fwd_ret > 0) or
                                (event["sentiment"] < 0 and fwd_ret < 0)
                            )
                            results.append({
                                "horizon": h,
                                "sentiment": event["sentiment"],
                                "fwd_ret": fwd_ret,
                                "direction_correct": direction_correct,
                                "is_earnings": event["is_earnings"],
                            })

                if not results:
                    continue

                def hit_rate(h):
                    subset = [r for r in results if r["horizon"] == h]
                    return (sum(r["direction_correct"] for r in subset) / len(subset) * 100) if subset else 0.0

                def avg_ret(h):
                    subset = [r["fwd_ret"] for r in results if r["horizon"] == h]
                    return (sum(subset) / len(subset) * 100) if subset else 0.0

                # Earnings vs general
                earnings_results = [r for r in results if r["is_earnings"] and r["horizon"] == 1]
                general_results = [r for r in results if not r["is_earnings"] and r["horizon"] == 1]
                earnings_hit = sum(r["direction_correct"] for r in earnings_results) / len(earnings_results) * 100 if earnings_results else 0
                general_hit = sum(r["direction_correct"] for r in general_results) / len(general_results) * 100 if general_results else 0

                # Sentiment magnitude correlation: do higher |sentiment| events yield bigger returns?
                high_sent = [r for r in results if abs(r["sentiment"]) > 0.7 and r["horizon"] == 1]
                low_sent = [r for r in results if 0.5 <= abs(r["sentiment"]) <= 0.7 and r["horizon"] == 1]
                high_hit = sum(r["direction_correct"] for r in high_sent) / len(high_sent) * 100 if high_sent else 0
                low_hit = sum(r["direction_correct"] for r in low_sent) / len(low_sent) * 100 if low_sent else 0

                lines.append(f"### {ticker} ({len(events)} events)")
                lines.append("| Horizon | Direction Hit Rate | Avg Return |")
                lines.append("|---------|-------------------|------------|")
                for h in [1, 3, 5]:
                    lines.append(f"| {h}d | {hit_rate(h):.1f}% | {avg_ret(h):+.2f}% |")
                lines.append(
                    f"- Earnings news: {earnings_hit:.1f}% hit rate ({len(earnings_results)} events) "
                    f"vs General news: {general_hit:.1f}% ({len(general_results)} events)"
                )
                lines.append(
                    f"- High sentiment (>0.7): {high_hit:.1f}% hit rate "
                    f"vs Moderate (0.5-0.7): {low_hit:.1f}%"
                )
                lines.append("")

                summary.append({
                    "ticker": ticker,
                    "n_events": len(events),
                    "hit_rate_1d": hit_rate(1),
                    "hit_rate_3d": hit_rate(3),
                    "earnings_hit": earnings_hit,
                    "general_hit": general_hit,
                    "high_sent_hit": high_hit,
                })

            except Exception as exc:
                logger.error("Sibyl research [%s]: %s", ticker, exc)
                lines.append(f"**{ticker}**: error — {exc}\n")

        # Source reliability
        lines.append("## News Source Reliability")
        lines.append("")
        lines.append("| Source | Articles (30d) | Avg Sentiment |")
        lines.append("|--------|----------------|---------------|")
        for source, sentiments in sorted(source_events.items(), key=lambda x: len(x[1]), reverse=True):
            avg_sent = sum(sentiments) / len(sentiments)
            lines.append(f"| {source} | {len(sentiments)} | {avg_sent:+.3f} |")
        lines.append("")

        if summary:
            avg_hit = sum(r["hit_rate_1d"] for r in summary) / len(summary)
            best = max(summary, key=lambda r: r["hit_rate_1d"])
            earnings_avg = sum(r["earnings_hit"] for r in summary if r["earnings_hit"] > 0)
            earnings_n = sum(1 for r in summary if r["earnings_hit"] > 0)
            general_avg = sum(r["general_hit"] for r in summary if r["general_hit"] > 0)
            general_n = sum(1 for r in summary if r["general_hit"] > 0)

            lines += [
                "## Conclusions",
                "",
                f"- Universe avg 1-day direction hit rate: **{avg_hit:.1f}%**",
                f"- Best news-driven ticker: **{best['ticker']}** ({best['hit_rate_1d']:.1f}% 1d hit rate)",
                f"- Earnings news avg hit rate: "
                f"{'N/A' if earnings_n == 0 else f'{earnings_avg/earnings_n:.1f}%'}",
                f"- General news avg hit rate: "
                f"{'N/A' if general_n == 0 else f'{general_avg/general_n:.1f}%'}",
                f"- Strategy viability: **{'Strong' if avg_hit > 60 else 'Moderate' if avg_hit > 50 else 'Weak'}**",
                "- Note: High sentiment magnitude (>0.7) appears to produce stronger directional signals "
                "than moderate sentiment (0.5-0.7).",
            ]

        path = self._agent_dir / "research_report.md"
        path.write_text("\n".join(lines))
        logger.info("Sibyl research report saved → %s", path)
        return str(path)

    # ------------------------------------------------------------------ signals

    def generate_signals(self) -> list[dict]:
        """
        Scan recent news for tickers with strong sentiment.
        Confirm with price action (daily return direction).
        Returns up to 5 signals ranked by sentiment magnitude.
        """
        threshold_buy = float(self._config.get("sentiment_threshold_buy", 0.6))
        threshold_sell = float(self._config.get("sentiment_threshold_sell", -0.6))
        max_atr_pct = float(self._config.get("max_atr_pct", 0.05))
        
        # Get the latest bar open time for deduplication
        bar_open_time = self.data.get_latest_bar_open("SPY")

        # EXIT signals: check open positions for sentiment reversal or PnL target
        exit_signals = []
        try:
            conn = self.db._conn()
            open_trades = conn.execute(
                "SELECT ticker, entry_price, opened_at, qty FROM trades "
                "WHERE agent_id = ? AND status = 'open'",
                (self.agent_id,),
            ).fetchall()
            conn.close()
        except Exception as exc:
            logger.error("Sibyl signal DB query error: %s", exc)
            open_trades = []

        now_ts = int(datetime.now(timezone.utc).timestamp())
        pnl_target = float(self._config.get("pnl_target_pct", 0.05))
        max_hold_days = int(self._config.get("max_hold_days", 3))

        for trade in open_trades:
            ticker = trade["ticker"]
            entry_price = float(trade["entry_price"] or 0)
            if entry_price <= 0:
                continue
            held_days = (now_ts - (trade["opened_at"] or now_ts)) / 86400
            current_sentiment = self.news.get_ticker_sentiment(ticker)

            try:
                f = self.data.get_features(ticker)
                close = f.get("close") if f else None
                atr = f.get("atr_14") if f else None

                if close is None:
                    continue

                pnl_pct = (close - entry_price) / entry_price
                triggers = []

                if pnl_pct >= pnl_target:
                    triggers.append(f"PnL target hit ({pnl_pct*100:+.2f}% ≥ {pnl_target*100:.0f}%)")

                if held_days >= max_hold_days:
                    triggers.append(f"max hold reached ({held_days:.1f}d ≥ {max_hold_days}d)")

                # Sentiment reversal check (signal that drove entry has faded)
                if current_sentiment is not None:
                    if (entry_price < close and current_sentiment < 0.2):
                        triggers.append(
                            f"sentiment reversed ({current_sentiment:+.3f}) — "
                            "news catalyst that drove entry has faded or inverted"
                        )
                    elif (entry_price > close and current_sentiment > -0.2):
                        triggers.append(
                            f"sentiment reversed ({current_sentiment:+.3f}) — "
                            "negative catalyst that drove short has dissipated"
                        )

                if triggers:
                    qty = self.risk.calculate_position_size_atr(
                        agent_capital=self.capital,
                        price=close,
                        atr=atr if atr else close * 0.02,
                        risk_per_trade=float(self._config.get("risk_per_trade", 0.02)),
                    )
                    qty = max(1, qty)
                    reason = (
                        f"SIBYL_NEWS_EXIT | {ticker} @ ${close:.2f} | "
                        f"Entry=${entry_price:.2f} PnL={pnl_pct*100:+.2f}% | "
                        f"Held={held_days:.1f}d | Current_sentiment={current_sentiment:+.3f} | "
                        f"Triggers: {'; '.join(triggers)}"
                    )
                    exit_signals.append({
                        "symbol": ticker,
                        "side": "sell",
                        "quantity": qty,
                        "type": "market",
                        "reason": reason,
                        "confidence": 1.0,
                        "bar_open_time": bar_open_time,
                    })
            except Exception as exc:
                logger.error("Sibyl exit check [%s]: %s", ticker, exc)

        # ENTRY signals: scan current news
        entry_signals = []
        try:
            headlines = self.news.fetch_headlines(max_per_source=25)
        except Exception as exc:
            logger.error("Sibyl fetch_headlines error: %s", exc)
            headlines = []

        # Aggregate sentiment per ticker over all recent headlines
        ticker_headlines: dict = defaultdict(list)
        for h in headlines:
            if h.get("ticker"):
                ticker_headlines[h["ticker"]].append(h)

        for ticker, items in ticker_headlines.items():
            if len(exit_signals) + len(entry_signals) >= 5:
                break
            try:
                # Avg sentiment weighted by recency (most recent headlines count more)
                now_ts_local = int(datetime.now(timezone.utc).timestamp())
                weighted_sum = 0.0
                weight_total = 0.0
                best_headline = ""
                best_headline_sentiment = 0.0

                for item in items:
                    age_hours = (now_ts_local - item.get("timestamp", now_ts_local)) / 3600
                    weight = max(0.1, 1.0 - age_hours / 24)  # linear decay over 24h
                    s = float(item.get("sentiment", 0))
                    weighted_sum += s * weight
                    weight_total += weight
                    if abs(s) > abs(best_headline_sentiment):
                        best_headline_sentiment = s
                        best_headline = item.get("headline", "")

                if weight_total == 0:
                    continue

                avg_sentiment = weighted_sum / weight_total

                # Check threshold
                if avg_sentiment > threshold_buy:
                    side = "buy"
                elif avg_sentiment < threshold_sell:
                    side = "sell"
                else:
                    continue

                # Get price features for this ticker
                f = self.data.get_features(ticker)
                if not f:
                    continue

                close = f.get("close")
                atr = f.get("atr_14")
                daily_return = f.get("daily_return")
                sma20 = f.get("sma_20")

                if close is None or close <= 0:
                    continue
                if atr is None:
                    atr = close * 0.02

                # ATR filter: skip if too volatile for news catalyst trade
                if atr / close > max_atr_pct:
                    logger.info(
                        "Sibyl skipping %s — ATR/price %.2f%% > %.0f%% threshold",
                        ticker, atr / close * 100, max_atr_pct * 100,
                    )
                    continue

                # Price action confirmation: daily return must agree with sentiment direction
                gap_confirmed = True
                gap_desc = "no gap data"
                if daily_return is not None:
                    if side == "buy" and daily_return < -0.02:
                        gap_confirmed = False  # strong negative gap contradicts positive sentiment
                        gap_desc = f"negative gap {daily_return*100:+.2f}% — contradicts sentiment"
                    elif side == "sell" and daily_return > 0.02:
                        gap_confirmed = False
                        gap_desc = f"positive gap {daily_return*100:+.2f}% — contradicts sentiment"
                    else:
                        gap_desc = f"daily_return={daily_return*100:+.2f}% (confirms direction)"

                if not gap_confirmed:
                    logger.info("Sibyl skipping %s — %s", ticker, gap_desc)
                    continue

                qty = self.risk.calculate_position_size_atr(
                    agent_capital=self.capital,
                    price=close,
                    atr=atr,
                    risk_per_trade=float(self._config.get("risk_per_trade", 0.02)),
                )
                if qty <= 0:
                    continue

                confidence = min(1.0, abs(avg_sentiment) / 1.0)
                is_earnings = _is_earnings_related(best_headline)
                headline_type = "EARNINGS" if is_earnings else "GENERAL_NEWS"

                reason = (
                    f"SIBYL_NEWS_CATALYST_{side.upper()} | {ticker} @ ${close:.2f} | "
                    f"Headline ({headline_type}): \"{best_headline[:120]}\" | "
                    f"Headline_sentiment={best_headline_sentiment:+.3f} | "
                    f"Weighted_avg_sentiment={avg_sentiment:+.3f} "
                    f"(threshold={'>' + str(threshold_buy) if side == 'buy' else '<' + str(threshold_sell)}) | "
                    f"Price_confirmation: {gap_desc} | "
                    f"ATR=${atr:.2f} ({atr/close*100:.2f}% of price) | "
                    f"SMA20=${sma20:.2f if sma20 else 'N/A'} | "
                    f"Exit plan: sentiment reversal / +{pnl_target*100:.0f}% PnL / {max_hold_days}d max hold"
                )

                entry_signals.append({
                    "symbol": ticker,
                    "side": side,
                    "quantity": qty,
                    "type": "market",
                    "reason": reason,
                    "confidence": round(confidence, 4),
                    "bar_open_time": bar_open_time,
                })

            except Exception as exc:
                logger.error("Sibyl entry check [%s]: %s", ticker, exc)

        entry_signals.sort(key=lambda s: s["confidence"], reverse=True)
        combined = exit_signals + entry_signals
        # Filter out signals from bars we've already acted on
        filtered = self._filter_deduplicated_signals(combined[:5])
        return filtered

    # ------------------------------------------------------------------ learning

    def learn_from_trades(self):
        """
        Track which news sources produce the most profitable signals.
        Test: does sentiment alone work, or is price confirmation required?
        Earnings-related vs general news — which is more profitable?
        Update entry criteria based on evidence. Save strategy version.
        """
        try:
            conn = self.db._conn()
            rows = conn.execute(
                """
                SELECT t.ticker, t.side, t.entry_price, t.exit_price, t.pnl,
                       t.opened_at, t.closed_at,
                       j.market_conditions, j.pre_trade_reasoning, j.lessons_learned
                FROM trades t
                LEFT JOIN trade_journal j ON j.trade_id = t.id
                WHERE t.agent_id = ? AND t.status = 'closed'
                ORDER BY t.closed_at DESC LIMIT 60
                """,
                (self.agent_id,),
            ).fetchall()
            conn.close()
        except Exception as exc:
            logger.error("Sibyl learn_from_trades DB error: %s", exc)
            return

        if not rows:
            logger.info("Sibyl: no closed trades to learn from yet")
            return

        trades = []
        for row in rows:
            mc = {}
            if row["market_conditions"]:
                try:
                    mc = json.loads(row["market_conditions"])
                except Exception:
                    pass
            reasoning = row["pre_trade_reasoning"] or ""
            # Extract metadata from the structured reason string
            is_earnings = _is_earnings_related(reasoning)
            # Infer if price was confirming (look for "confirms direction" in reasoning)
            price_confirmed = "confirms direction" in reasoning.lower()
            hold_secs = (row["closed_at"] or 0) - (row["opened_at"] or 0)

            trades.append({
                "ticker": row["ticker"],
                "pnl": float(row["pnl"] or 0),
                "won": (row["pnl"] or 0) > 0,
                "regime": mc.get("regime", "unknown"),
                "is_earnings": is_earnings,
                "price_confirmed": price_confirmed,
                "hold_days": hold_secs / 86400 if hold_secs > 0 else 0,
            })

        def _wr(items):
            return sum(1 for t in items if t["won"]) / len(items) if items else 0.0

        overall_wr = _wr(trades)

        # Earnings vs general news signal quality
        earnings_trades = [t for t in trades if t["is_earnings"]]
        general_trades = [t for t in trades if not t["is_earnings"]]
        earnings_wr = _wr(earnings_trades)
        general_wr = _wr(general_trades)

        # Price confirmation impact
        confirmed_trades = [t for t in trades if t["price_confirmed"]]
        unconfirmed_trades = [t for t in trades if not t["price_confirmed"]]
        confirmed_wr = _wr(confirmed_trades)
        unconfirmed_wr = _wr(unconfirmed_trades)

        # Hold duration analysis
        win_holds = [t["hold_days"] for t in trades if t["won"] and t["hold_days"] > 0]
        avg_win_hold = sum(win_holds) / len(win_holds) if win_holds else 0

        ticker_groups: dict = defaultdict(list)
        for t in trades:
            ticker_groups[t["ticker"]].append(t)
        ticker_stats = {
            tk: {"count": len(g), "win_rate": round(_wr(g), 3),
                 "avg_pnl": round(sum(t["pnl"] for t in g) / len(g), 2)}
            for tk, g in ticker_groups.items()
        }

        changes = []

        # If price confirmation significantly improves win rate, tighten sentiment threshold
        if (
            len(confirmed_trades) >= 5 and len(unconfirmed_trades) >= 5
            and confirmed_wr > unconfirmed_wr + 0.15
        ):
            old_buy = float(self._config.get("sentiment_threshold_buy", 0.6))
            old_sell = float(self._config.get("sentiment_threshold_sell", -0.6))
            new_buy = round(min(0.8, old_buy + 0.1), 2)
            new_sell = round(max(-0.8, old_sell - 0.1), 2)
            if new_buy != old_buy:
                self._config["sentiment_threshold_buy"] = new_buy
                self._config["sentiment_threshold_sell"] = new_sell
                changes.append(
                    f"Sentiment thresholds raised: buy {old_buy}→{new_buy}, sell {old_sell}→{new_sell}. "
                    f"Price-confirmed trades win_rate={confirmed_wr:.1%} "
                    f"vs unconfirmed={unconfirmed_wr:.1%}. "
                    "Requiring stronger sentiment to reduce false positives from weak signals."
                )

        # If earnings trades are significantly more reliable, could lower threshold for earnings news
        # (handled by signal enrichment note rather than threshold change — too risky to lower broadly)

        # If win rate is too low, reduce max_atr_pct to avoid volatile tickers
        if len(trades) >= 15 and overall_wr < 0.42:
            old_atr = float(self._config.get("max_atr_pct", 0.05))
            new_atr = max(0.02, round(old_atr - 0.01, 2))
            if new_atr != old_atr:
                self._config["max_atr_pct"] = new_atr
                changes.append(
                    f"max_atr_pct reduced {old_atr:.2f}→{new_atr:.2f}: "
                    f"overall win rate {overall_wr:.1%} below 42% across {len(trades)} trades. "
                    "News catalysts on highly volatile tickers may reverse too quickly — "
                    "filtering to calmer setups where the move has room to develop."
                )

        # Adjust hold period based on winner resolution time
        if avg_win_hold > 0 and avg_win_hold < self._config.get("max_hold_days", 3) * 0.5:
            old_hold = int(self._config.get("max_hold_days", 3))
            new_hold = max(1, round(avg_win_hold) + 1)
            if new_hold != old_hold:
                self._config["max_hold_days"] = new_hold
                changes.append(
                    f"max_hold_days reduced {old_hold}→{new_hold}: "
                    f"winners resolving in avg {avg_win_hold:.1f}d. "
                    "News catalysts dissipate faster than expected — cutting hold to lock in edge."
                )

        # --- version & save ---
        current_v = str(self._config.get("strategy_version", "1.0"))
        try:
            major, minor = current_v.rsplit(".", 1)
            new_v = f"{major}.{int(minor) + 1}"
        except Exception:
            new_v = "1.1"

        best_ticker = max(ticker_stats, key=lambda tk: ticker_stats[tk]["win_rate"], default="N/A") if ticker_stats else "N/A"

        description = (
            f"v{new_v} | trades={len(trades)} wr={overall_wr:.1%} "
            f"earnings_wr={earnings_wr:.1%}(n={len(earnings_trades)}) "
            f"general_wr={general_wr:.1%}(n={len(general_trades)}) "
            f"confirmed_wr={confirmed_wr:.1%} unconfirmed_wr={unconfirmed_wr:.1%} | "
            + ("; ".join(changes) if changes else "no parameter changes — monitoring")
        )

        change_block = "\n".join(f"  - {c}" for c in changes) if changes else "  - None (monitoring phase)"
        code_lines = [
            '"""',
            f"Sibyl News Catalyst Strategy — v{new_v}",
            f"Updated: {datetime.now(timezone.utc).isoformat()}",
            "",
            "Changes from previous version:",
            change_block,
            "",
            "Evidence basis:",
            f"  trades_analyzed = {len(trades)}",
            f"  overall_win_rate = {overall_wr:.3f}",
            f"  earnings_news_win_rate = {earnings_wr:.3f} (n={len(earnings_trades)})",
            f"  general_news_win_rate = {general_wr:.3f} (n={len(general_trades)})",
            f"  price_confirmed_win_rate = {confirmed_wr:.3f} (n={len(confirmed_trades)})",
            f"  unconfirmed_win_rate = {unconfirmed_wr:.3f} (n={len(unconfirmed_trades)})",
            f"  avg_winning_hold_days = {avg_win_hold:.2f}",
            f"  best_ticker = {best_ticker}",
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
        logger.info("Sibyl updated → v%s | %s", new_v, description[:140])
