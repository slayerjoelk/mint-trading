import logging
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf
import ta  # technical analysis — installed: 'ta' package
from scipy import stats  # for statistical tests

logger = logging.getLogger(__name__)


class DataPipeline:

    # ── Raw Data ──────────────────────────────────────────────────────────────

    def get_daily_bars(self, ticker: str, period: str = "3mo") -> Optional[pd.DataFrame]:
        try:
            df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
            if df is None or df.empty:
                logger.warning("No data for %s", ticker)
                return None
            df.columns = [c.lower() if isinstance(c, str) else c[0].lower() for c in df.columns]
            df = df[["open", "high", "low", "close", "volume"]].copy()
            df.dropna(inplace=True)
            return df
        except Exception as e:
            logger.error("get_daily_bars error [%s]: %s", ticker, e)
            return None

    # ── Indicators ────────────────────────────────────────────────────────────

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all technical indicators using the 'ta' library."""
        df = df.copy()
        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]

        # Trend indicators
        df["sma_20"] = ta.trend.sma_indicator(close, window=20)
        df["sma_50"] = ta.trend.sma_indicator(close, window=50)
        df["ema_12"] = ta.trend.ema_indicator(close, window=12)
        df["ema_26"] = ta.trend.ema_indicator(close, window=26)

        # MACD
        macd = ta.trend.MACD(close)
        df["macd"] = macd.macd()
        df["macd_signal"] = macd.macd_signal()
        df["macd_hist"] = macd.macd_diff()

        # Momentum
        df["rsi_14"] = ta.momentum.rsi(close, window=14)

        # Volatility — Bollinger Bands
        bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
        df["bb_upper"] = bb.bollinger_hband()
        df["bb_mid"] = bb.bollinger_mavg()
        df["bb_lower"] = bb.bollinger_lband()
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]

        # ATR
        df["atr_14"] = ta.volatility.average_true_range(high, low, close, window=14)

        # Volume
        df["vol_sma_20"] = ta.volume.volume_weighted_average_price(high, low, close, volume, window=20)
        # Note: VWAP is close-proxy for volume trend. Real vol SMA:
        df["vol_sma_20"] = volume.rolling(20).mean()

        return df

    # ── Features ──────────────────────────────────────────────────────────────

    def get_features(self, ticker: str) -> dict:
        df = self.get_daily_bars(ticker)
        if df is None or len(df) < 60:
            logger.warning("Insufficient data for features [%s]", ticker)
            return {}
        df = self.compute_indicators(df)
        latest = df.iloc[-1]
        prev = df.iloc[-2]

        def _f(val):
            return float(val) if pd.notna(val) else None

        return {
            "ticker": ticker,
            "close": _f(latest["close"]),
            "sma_20": _f(latest["sma_20"]),
            "sma_50": _f(latest["sma_50"]),
            "ema_12": _f(latest["ema_12"]),
            "ema_26": _f(latest["ema_26"]),
            "macd": _f(latest["macd"]),
            "macd_signal": _f(latest["macd_signal"]),
            "macd_hist": _f(latest["macd_hist"]),
            "rsi_14": _f(latest["rsi_14"]),
            "bb_upper": _f(latest["bb_upper"]),
            "bb_mid": _f(latest["bb_mid"]),
            "bb_lower": _f(latest["bb_lower"]),
            "bb_width": _f(latest["bb_width"]),
            "atr_14": _f(latest["atr_14"]),
            "volume": _f(latest["volume"]),
            "vol_sma_20": _f(latest["vol_sma_20"]),
            "vol_ratio": _f(latest["volume"] / latest["vol_sma_20"]) if latest["vol_sma_20"] else None,
            "price_vs_sma20": _f((latest["close"] - latest["sma_20"]) / latest["sma_20"]) if latest["sma_20"] else None,
            "price_vs_sma50": _f((latest["close"] - latest["sma_50"]) / latest["sma_50"]) if latest["sma_50"] else None,
            "daily_return": _f((latest["close"] - prev["close"]) / prev["close"]),
        }

    # ── Market Regime ─────────────────────────────────────────────────────────

    def get_market_regime(self, ticker: str = "SPY") -> str:
        try:
            df = self.get_daily_bars(ticker, period="6mo")
            if df is None or len(df) < 60:
                return "unknown"
            df = self.compute_indicators(df)
            latest = df.iloc[-1]

            close = float(latest["close"])
            sma_20 = float(latest["sma_20"]) if pd.notna(latest["sma_20"]) else None
            sma_50 = float(latest["sma_50"]) if pd.notna(latest["sma_50"]) else None
            atr_14 = float(latest["atr_14"]) if pd.notna(latest["atr_14"]) else None
            rsi = float(latest["rsi_14"]) if pd.notna(latest["rsi_14"]) else None

            if sma_20 is None or sma_50 is None or atr_14 is None:
                return "unknown"

            atr_pct = atr_14 / close
            trend_diff = (sma_20 - sma_50) / sma_50

            if atr_pct > 0.025:
                return "high_volatility"
            if atr_pct < 0.008:
                return "low_volatility"
            if trend_diff > 0.02 and close > sma_20 > sma_50:
                return "trending_up"
            if trend_diff < -0.02 and close < sma_20 < sma_50:
                return "trending_down"
            return "ranging"
        except Exception as e:
            logger.error("get_market_regime error: %s", e)
            return "unknown"

    # ── Universe Stats ────────────────────────────────────────────────────────

    def get_universe_stats(self, tickers: list[str]) -> pd.DataFrame:
        rows = []
        for ticker in tickers:
            try:
                df = self.get_daily_bars(ticker, period="3mo")
                if df is None or len(df) < 21:
                    rows.append({"ticker": ticker, "momentum": None, "volatility": None, "volume_ratio": None})
                    continue

                df = self.compute_indicators(df)
                latest = df.iloc[-1]
                month_ago = df.iloc[-21]

                momentum = (float(latest["close"]) - float(month_ago["close"])) / float(month_ago["close"])
                volatility = float(df["close"].pct_change().tail(21).std()) * (252 ** 0.5)
                vol_ratio = (float(latest["volume"]) / float(latest["vol_sma_20"])
                             if pd.notna(latest["vol_sma_20"]) and latest["vol_sma_20"] > 0 else None)

                rows.append({
                    "ticker": ticker,
                    "momentum": round(momentum, 4),
                    "volatility": round(volatility, 4),
                    "volume_ratio": round(vol_ratio, 4) if vol_ratio is not None else None,
                })
            except Exception as e:
                logger.error("get_universe_stats error [%s]: %s", ticker, e)
                rows.append({"ticker": ticker, "momentum": None, "volatility": None, "volume_ratio": None})

        return pd.DataFrame(rows).set_index("ticker")
