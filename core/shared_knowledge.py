import json
import logging
import uuid
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

VALID_CATEGORIES = {
    "regime", "ticker_behavior", "risk_parameter", "entry_timing", "exit_timing",
    "sector_rotation", "correlation", "volatility", "news_impact",
}


def _now() -> int:
    return int(datetime.now(timezone.utc).timestamp())


class SharedKnowledge:
    def __init__(self, db):
        self.db = db

    def add_insight(self, agent_id: str, agent_name: str, category: str,
                    insight: str, evidence: dict, confidence: float,
                    created_at: int = None, ticker: str = None) -> str:
        insight_id = str(uuid.uuid4())
        self.db.insert_shared_knowledge(
            id=insight_id,
            agent_id=agent_id,
            agent_name=agent_name,
            category=category,
            insight=insight,
            evidence=evidence,
            confidence=min(1.0, max(0.0, confidence)),
            ticker=ticker,
            created_at=created_at or _now(),
        )
        return insight_id

    def query(self, category: str = None, ticker: str = None,
              min_confidence: float = 0.5) -> list[dict]:
        try:
            conn = self.db._conn()
            clauses = ["confidence >= ?"]
            params: list = [min_confidence]

            if category:
                clauses.append("category = ?")
                params.append(category)
            if ticker:
                clauses.append("(ticker = ? OR ticker IS NULL)")
                params.append(ticker)

            sql = f"""
            SELECT id, agent_id, agent_name, category, insight, evidence,
                   confidence, ticker, created_at
            FROM shared_knowledge
            WHERE {' AND '.join(clauses)}
            ORDER BY confidence DESC, created_at DESC
            LIMIT 200
            """
            rows = conn.execute(sql, params).fetchall()
            conn.close()
            results = []
            for r in rows:
                ev = None
                if r["evidence"]:
                    try:
                        ev = json.loads(r["evidence"])
                    except Exception:
                        ev = r["evidence"]
                results.append({
                    "id": r["id"],
                    "agent_id": r["agent_id"],
                    "agent_name": r["agent_name"],
                    "category": r["category"],
                    "insight": r["insight"],
                    "evidence": ev,
                    "confidence": r["confidence"],
                    "ticker": r["ticker"],
                    "created_at": r["created_at"],
                })
            return results
        except Exception as exc:
            logger.error("SharedKnowledge.query error: %s", exc)
            return []

    def get_regime_playbook(self, regime: str) -> dict:
        insights = self.query(category="regime", min_confidence=0.5)
        relevant = [i for i in insights if regime.lower() in i["insight"].lower()]

        strategies: dict[str, list] = {}
        tickers: list[str] = []
        risk_notes: list[str] = []

        for i in relevant:
            agent = i["agent_name"]
            strategies.setdefault(agent, []).append({
                "insight": i["insight"],
                "confidence": i["confidence"],
                "evidence": i["evidence"],
            })
            if i.get("ticker"):
                tickers.append(i["ticker"])

        risk_insights = self.query(category="risk_parameter", min_confidence=0.5)
        for i in risk_insights:
            if regime.lower() in i["insight"].lower():
                risk_notes.append(i["insight"])

        return {
            "regime": regime,
            "strategies_by_agent": strategies,
            "relevant_tickers": list(set(tickers)),
            "risk_notes": risk_notes,
            "total_insights": len(relevant),
        }

    def get_ticker_consensus(self, ticker: str) -> dict:
        insights = self.query(ticker=ticker, min_confidence=0.3)
        if not insights:
            return {
                "ticker": ticker,
                "avg_confidence": 0.0,
                "buy_signals": 0,
                "sell_signals": 0,
                "hold_signals": 0,
                "consensus": "neutral",
                "insights": [],
            }

        buy_signals = 0
        sell_signals = 0
        hold_signals = 0
        confidences = []

        for i in insights:
            text = i["insight"].lower()
            conf = i["confidence"]
            confidences.append(conf)
            if any(w in text for w in ("buy", "long", "oversold", "revert up", "bullish")):
                buy_signals += 1
            elif any(w in text for w in ("sell", "short", "overbought", "revert down", "bearish")):
                sell_signals += 1
            else:
                hold_signals += 1

        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
        total = buy_signals + sell_signals + hold_signals

        if total == 0:
            consensus = "neutral"
        else:
            buy_pct = buy_signals / total
            sell_pct = sell_signals / total
            if buy_pct >= 0.7:
                consensus = "strong_buy"
            elif buy_pct >= 0.5:
                consensus = "buy"
            elif sell_pct >= 0.7:
                consensus = "strong_sell"
            elif sell_pct >= 0.5:
                consensus = "sell"
            else:
                consensus = "neutral"

        return {
            "ticker": ticker,
            "avg_confidence": round(avg_conf, 4),
            "buy_signals": buy_signals,
            "sell_signals": sell_signals,
            "hold_signals": hold_signals,
            "consensus": consensus,
            "insights": insights[:10],
        }
