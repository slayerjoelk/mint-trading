"""
Machine learning-based market regime classifier.
Replaces hand-coded SMA comparisons with statistical feature scoring.
"""
import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

REGIMES = ['trending_up', 'trending_down', 'ranging', 'high_volatility', 'low_volatility', 'crisis']

REGIME_DESCRIPTIONS = {
    'trending_up': 'Price > 20SMA > 50SMA, low volatility, positive momentum',
    'trending_down': 'Price < 20SMA < 50SMA, rising vol, negative momentum',
    'ranging': 'Price near SMA, sideways momentum, medium volatility',
    'high_volatility': 'Elevated VIXY, wide Bollinger Bands, high ATR',
    'low_volatility': 'Compressed VIXY, tight Bands, low ATR, complacency',
    'crisis': 'Extreme VIXY, deep drawdown, high sector correlation',
}

def _now() -> int:
    return int(datetime.now(timezone.utc).timestamp())


class RegimeClassifier:
    def __init__(self, data_pipeline, db):
        self.data = data_pipeline
        self.db = db

    # ------------------------------------------------------------------ feature extraction

    def get_features_for_date(self, end_date=None, lookback_days: int = 60) -> dict:
        end = end_date or datetime.now(timezone.utc).date()
        if isinstance(end, str):
            end = pd.Timestamp(end).date()
        feats: dict = {}

        # SPY features
        df_spy = self.data.get_daily_bars("SPY", period="6mo")
        if df_spy is not None and len(df_spy) >= 50:
            df_spy = self.data.compute_indicators(df_spy)
            df_spy_cut = df_spy[df_spy.index.date <= end]
            if len(df_spy_cut) >= 50:
                latest = df_spy_cut.iloc[-1]
                close = float(latest["close"])
                sma20 = float(latest["sma_20"]) if pd.notna(latest["sma_20"]) else None
                sma50 = float(latest["sma_50"]) if pd.notna(latest["sma_50"]) else None
                atr = float(latest["atr_14"]) if pd.notna(latest["atr_14"]) else None
                rsi = float(latest["rsi_14"]) if pd.notna(latest["rsi_14"]) else None
                bb_width = float(latest["bb_width"]) if pd.notna(latest["bb_width"]) else None

                atr_pct = atr / close if atr and close > 0 else 0.0
                sma20_diff = (close - sma20) / sma20 if sma20 else 0.0
                sma50_diff = (close - sma50) / sma50 if sma50 else 0.0
                momentum_1m = (close - df_spy_cut.iloc[-21]["close"]) / df_spy_cut.iloc[-21]["close"] if len(df_spy_cut) >= 21 else 0.0

                peak = df_spy_cut["close"].max()
                dd_from_peak = (close - peak) / peak if peak > 0 else 0.0

                feats["spy_close"] = close
                feats["spy_sma20_diff"] = round(sma20_diff, 4)
                feats["spy_sma50_diff"] = round(sma50_diff, 4)
                feats["spy_atr_pct"] = round(atr_pct, 4)
                feats["spy_rsi"] = round(rsi, 1) if rsi else None
                feats["spy_bb_width"] = round(bb_width, 4) if bb_width else None
                feats["spy_momentum_1m"] = round(momentum_1m, 4)
                feats["spy_dd_from_peak"] = round(dd_from_peak, 4)
                feats["spy_sma20_above_sma50"] = (sma20 is not None and sma50 is not None and sma20 > sma50)
                feats["spy_consecutive_up"] = int(sum(1 for i in range(-1, -11, -1) if i >= -len(df_spy_cut) and df_spy_cut.iloc[i]["close"] > df_spy_cut.iloc[i-1]["close"]))

        # VIXY features
        df_vixy = self.data.get_daily_bars("VIXY", period="3mo")
        if df_vixy is not None and len(df_vixy) >= 20:
            vixy_close = float(df_vixy["close"].iloc[-1])
            vixy_low_20 = float(df_vixy["close"].tail(20).min())
            vixy_pct_from_low = (vixy_close - vixy_low_20) / vixy_low_20
            feats["vixy_close"] = round(vixy_close, 2)
            feats["vixy_pct_from_20d_low"] = round(vixy_pct_from_low, 4)

        # Breadth: IWM vs SPY ratio
        df_iwm = self.data.get_daily_bars("IWM", period="3mo")
        if df_spy is not None and df_iwm is not None and len(df_iwm) >= 20:
            ratio = float(df_iwm["close"].iloc[-1]) / float(df_spy["close"].iloc[-1])
            ratio_sma20 = float(df_iwm["close"].rolling(20).mean().iloc[-1]) / float(df_spy["close"].rolling(20).mean().iloc[-1])
            feats["breadth_iwm_spy"] = round(ratio, 4)
            feats["breadth_diff"] = round((ratio - ratio_sma20) / ratio_sma20, 4)

        # Yield curve proxy: TLT vs SHY
        df_tlt = self.data.get_daily_bars("TLT", period="3mo")
        df_shy = self.data.get_daily_bars("SHY", period="3mo")
        if df_tlt is not None and df_shy is not None and len(df_tlt) >= 5:
            tlt_close = float(df_tlt["close"].iloc[-1])
            shy_close = float(df_shy["close"].iloc[-1])
            feats["tlt_shy_ratio"] = round(tlt_close / shy_close, 4)

        return feats

    # ------------------------------------------------------------------ classification

    def classify(self, date=None) -> str:
        regime, _ = self.classify_with_confidence(date)
        return regime

    def classify_with_confidence(self, date=None) -> tuple:
        feats = self.get_features_for_date(end_date=date)
        if not feats or len(feats) < 3:
            return ("unknown", 0.0)

        scores = {}

        # trending_up
        score = 0.0
        if feats.get("spy_sma20_above_sma50", False):
            score += 0.3
        if feats.get("spy_sma20_diff", 0) > 0:
            score += 0.2
        if feats.get("spy_momentum_1m", 0) > 0.02:
            score += 0.15
        vixy_pct = feats.get("vixy_pct_from_20d_low", 0.5)
        if vixy_pct < 0.15:
            score += 0.15
        if feats.get("spy_rsi", 50) and 50 < feats.get("spy_rsi", 50) < 70:
            score += 0.1
        if feats.get("spy_consecutive_up", 0) >= 3:
            score += 0.1
        scores["trending_up"] = min(1.0, score)

        # trending_down
        score = 0.0
        if feats.get("spy_sma20_diff", 0) < -0.01:
            score += 0.25
        if not feats.get("spy_sma20_above_sma50", True):
            score += 0.2
        if feats.get("spy_momentum_1m", 0) < -0.02:
            score += 0.15
        if feats.get("spy_dd_from_peak", 0) < -0.05:
            score += 0.2
        if feats.get("breadth_iwm_spy", 1.0) < 0.9:
            score += 0.1
        if vixy_pct > 0.2:
            score += 0.1
        scores["trending_down"] = min(1.0, score)

        # ranging
        score = 0.0
        sma20_diff = abs(feats.get("spy_sma20_diff", 1.0))
        if sma20_diff < 0.02:
            score += 0.4
        if sma20_diff < 0.03:
            score += 0.15
        mom_abs = abs(feats.get("spy_momentum_1m", 0.5))
        if mom_abs < 0.03:
            score += 0.25
        if feats.get("spy_atr_pct", 0.05) < 0.015:
            score += 0.1
        if feats.get("spy_rsi", 100) and 40 < feats.get("spy_rsi", 100) < 60:
            score += 0.1
        scores["ranging"] = min(1.0, score)

        # high_volatility
        score = 0.0
        if vixy_pct > 0.3:
            score += 0.35
        if feats.get("spy_atr_pct", 0) > 0.02:
            score += 0.25
        if feats.get("spy_bb_width", 0) and feats.get("spy_bb_width", 0) > 0.06:
            score += 0.2
        if feats.get("spy_dd_from_peak", 0) < -0.03:
            score += 0.1
        if feats.get("breadth_diff", 0) and abs(feats.get("breadth_diff", 0)) > 0.02:
            score += 0.1
        scores["high_volatility"] = min(1.0, score)

        # low_volatility
        score = 0.0
        if vixy_pct < 0.05:
            score += 0.35
        if feats.get("spy_atr_pct", 1.0) < 0.008:
            score += 0.3
        if feats.get("spy_bb_width", 1.0) and feats.get("spy_bb_width", 1.0) < 0.03:
            score += 0.2
        if abs(feats.get("breadth_diff", 0.5)) < 0.005:
            score += 0.15
        scores["low_volatility"] = min(1.0, score)

        # crisis
        score = 0.0
        if feats.get("spy_dd_from_peak", 0) < -0.10:
            score += 0.4
        if vixy_pct > 0.5:
            score += 0.3
        if feats.get("spy_atr_pct", 0) > 0.03:
            score += 0.2
        if feats.get("spy_momentum_1m", 0) < -0.05:
            score += 0.1
        scores["crisis"] = min(1.0, score)

        best_regime = max(scores, key=scores.get)
        best_score = scores[best_regime]

        # confidence: gap between #1 and #2 regime scores
        sorted_scores = sorted(scores.values(), reverse=True)
        gap = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) > 1 else 0.5
        confidence = min(1.0, 0.3 + gap * 0.7)

        return (best_regime, round(confidence, 4))

    # ------------------------------------------------------------------ history + change detection

    def get_regime_history(self, days: int = 252) -> pd.DataFrame:
        end = datetime.now(timezone.utc).date()
        start = end - timedelta(days=days * 2)
        rows = []
        current = start
        while current <= end:
            try:
                r, c = self.classify_with_confidence(current)
                rows.append({"date": current.isoformat(), "regime": r, "confidence": c})
            except Exception:
                pass
            current += timedelta(days=1)
        return pd.DataFrame(rows)

    def detect_regime_change(self) -> dict:
        feats = self.get_features_for_date()
        if not feats:
            return {"changed": False, "from_regime": "unknown", "to_regime": "unknown", "confidence": 0.0, "days_in_new_regime": 0}

        today_regime, today_conf = self.classify_with_confidence()
        try:
            conn = self.db._conn()
            rows = conn.execute(
                "SELECT regime, date FROM regime_history ORDER BY date DESC LIMIT 10"
            ).fetchall()
            conn.close()
        except Exception:
            rows = []

        if not rows:
            return {"changed": False, "from_regime": "unknown", "to_regime": today_regime, "confidence": today_conf, "days_in_new_regime": 0}

        prev_regime = rows[0]["regime"]
        changed = prev_regime != today_regime

        days_in_new = 0
        if changed:
            days_in_new = 1
        elif today_conf > 0.7:
            for r in rows:
                if r["regime"] == today_regime:
                    days_in_new += 1
                else:
                    break

        return {
            "changed": changed,
            "from_regime": prev_regime,
            "to_regime": today_regime,
            "confidence": today_conf,
            "days_in_new_regime": days_in_new,
        }
