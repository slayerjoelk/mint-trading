"""Alternative data pipeline — supplements price data with options flow, sector rotation, and breadth signals."""
import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_SECTOR_ETFS = {
    "XLK": "Technology", "XLF": "Financials", "XLE": "Energy", "XLV": "Healthcare",
    "XLI": "Industrials", "XLY": "Consumer Disc", "XLRE": "Real Estate",
    "XLU": "Utilities", "XLB": "Materials", "XLP": "Consumer Staples",
    "SMH": "Semiconductors", "XBI": "Biotech", "XRT": "Retail", "KRE": "Regional Banks",
}


class AlternativeData:
    def __init__(self, data_pipeline, db):
        self.data = data_pipeline
        self.db = db

    # ── options flow ─────────────────────────────────────────────────────────

    def get_options_flow(self, ticker: str) -> dict:
        try:
            import yfinance as yf
            stock = yf.Ticker(ticker)
            opts = stock.options
            if not opts:
                return {"put_call_ratio": None, "open_interest": None, "unusual_activity_score": 0.0}

            # Pull nearest expiry
            nearest = opts[0]
            chain = stock.option_chain(nearest)
            calls = chain.calls
            puts = chain.puts

            call_oi = float(calls["openInterest"].sum()) if "openInterest" in calls.columns else 0
            put_oi = float(puts["openInterest"].sum()) if "openInterest" in puts.columns else 0
            pcr = round(put_oi / call_oi, 4) if call_oi > 0 else None

            call_vol = float(calls["volume"].sum()) if "volume" in calls.columns else 0
            put_vol = float(puts["volume"].sum()) if "volume" in puts.columns else 0
            total_vol = call_vol + put_vol
            avg_vol = total_vol / 20 if total_vol > 0 else 0

            unusual_score = 0.0
            if pcr is not None:
                if pcr < 0.5:
                    unusual_score = 0.6  # bullish — calls dominate
                elif pcr > 2.0:
                    unusual_score = -0.6  # bearish — puts dominate

            return {
                "put_call_ratio": pcr,
                "call_oi": round(call_oi),
                "put_oi": round(put_oi),
                "unusual_activity_score": round(unusual_score, 2),
            }
        except Exception as e:
            logger.debug("options_flow [%s]: %s", ticker, e)
            return {"put_call_ratio": None, "unusual_activity_score": 0.0}

    # ── sector rotation ──────────────────────────────────────────────────────

    def get_sector_rotation(self) -> dict:
        momentum = {}
        for sym, name in _SECTOR_ETFS.items():
            df = self.data.get_daily_bars(sym, period="3mo")
            if df is None or len(df) < 21:
                continue
            week_ago = df["close"].iloc[-5] if len(df) >= 5 else df["close"].iloc[-1]
            month_ago = df["close"].iloc[-21] if len(df) >= 21 else df["close"].iloc[-1]
            latest = float(df["close"].iloc[-1])
            week_ret = (latest - float(week_ago)) / float(week_ago) if float(week_ago) > 0 else 0.0
            month_ret = (latest - float(month_ago)) / float(month_ago) if float(month_ago) > 0 else 0.0
            momentum[sym] = {"name": name, "week_ret": round(week_ret, 4), "month_ret": round(month_ret, 4)}

        if not momentum:
            return {"top_sectors": [], "bottom_sectors": [], "rotation_signal": "unknown"}

        sorted_by_month = sorted(momentum.items(), key=lambda x: x[1]["month_ret"], reverse=True)
        top = [{"ticker": s, "name": d["name"], "month_ret": d["month_ret"]} for s, d in sorted_by_month[:3]]
        bottom = [{"ticker": s, "name": d["name"], "month_ret": d["month_ret"]} for s, d in sorted_by_month[-3:]]

        top_types = [d["name"] for _, d in sorted_by_month[:3]]
        rotation_signal = "growth" if "Technology" in top_types or "Consumer Disc" in top_types else "defensive"

        return {
            "top_sectors": top,
            "bottom_sectors": bottom,
            "rotation_signal": rotation_signal,
            "leader": top[0]["name"] if top else "unknown",
            "laggard": bottom[0]["name"] if bottom else "unknown",
        }

    # ── market breadth ───────────────────────────────────────────────────────

    def get_market_breadth(self) -> dict:
        df_spy = self.data.get_daily_bars("SPY", period="3mo")
        df_iwm = self.data.get_daily_bars("IWM", period="3mo")
        df_rsp = self.data.get_daily_bars("RSP", period="3mo")

        breadth_health = "unknown"
        participation_score = 0.5

        if df_spy is not None and df_iwm is not None and len(df_spy) >= 20 and len(df_iwm) >= 20:
            spy_close = float(df_spy["close"].iloc[-1])
            iwm_close = float(df_iwm["close"].iloc[-1])
            spy_sma20 = float(df_spy["sma_20"].iloc[-1]) if "sma_20" in df_spy.columns and pd.notna(df_spy["sma_20"].iloc[-1]) else spy_close
            iwm_sma20 = float(df_iwm["sma_20"].iloc[-1]) if "sma_20" in df_iwm.columns and pd.notna(df_iwm["sma_20"].iloc[-1]) else iwm_close

            spy_above = spy_close > spy_sma20
            iwm_above = iwm_close > iwm_sma20
            if spy_above and iwm_above:
                breadth_health = "broad_participation"
                participation_score = 0.8
            elif spy_above and not iwm_above:
                breadth_health = "narrow_rally"
                participation_score = 0.3
            elif not spy_above and not iwm_above:
                breadth_health = "broad_decline"
                participation_score = 0.2
            else:
                breadth_health = "divergent"
                participation_score = 0.4

        divergence_warning = participation_score < 0.35
        return {
            "breadth_health": breadth_health,
            "participation_score": participation_score,
            "divergence_warning": divergence_warning,
        }

    # ── economic surprise proxy ──────────────────────────────────────────────

    def get_economic_surprise_proxy(self) -> float:
        df_tlt = self.data.get_daily_bars("TLT", period="3mo")
        df_shy = self.data.get_daily_bars("SHY", period="3mo")
        if df_tlt is None or df_shy is None or len(df_tlt) < 20 or len(df_shy) < 20:
            return 0.0

        tlt_closes = df_tlt["close"].values
        shy_closes = df_shy["close"].values
        ratios = tlt_closes[-20:] / shy_closes[-20:]
        ratio_now = ratios[-1]
        ratio_mean = float(np.mean(ratios))
        ratio_z = (ratio_now - ratio_mean) / float(np.std(ratios)) if np.std(ratios) > 0 else 0.0

        # Higher TLT/SHY = bond market pricing in slower growth / recession risk → negative surprise
        # Lower TLT/SHY = pricing expansion → positive surprise
        growth_score = -np.clip(ratio_z, -2, 2) / 2
        return round(float(growth_score), 4)
