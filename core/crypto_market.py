"""Crypto market interface via ccxt. Falls back to mock data when no API keys configured."""
import logging
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import ccxt
    _CCXT_AVAILABLE = True
except ImportError:
    _CCXT_AVAILABLE = False


class CryptoMarket:
    def __init__(self, exchange_id: str = "binance", api_key: str = None, secret: str = None):
        self.exchange_id = exchange_id
        self.has_keys = bool(api_key and secret)
        self._exchange = None
        if _CCXT_AVAILABLE:
            try:
                exch_class = getattr(ccxt, exchange_id)
                self._exchange = exch_class({
                    "apiKey": api_key or "",
                    "secret": secret or "",
                    "enableRateLimit": True,
                })
            except Exception as e:
                logger.warning("ccxt exchange init failed: %s", e)

    def get_ticker(self, symbol: str) -> dict:
        if self._exchange:
            try:
                t = self._exchange.fetch_ticker(symbol)
                return {
                    "symbol": symbol,
                    "last": float(t.get("last", 0)),
                    "bid": float(t.get("bid", 0)),
                    "ask": float(t.get("ask", 0)),
                    "high_24h": float(t.get("high", 0)),
                    "low_24h": float(t.get("low", 0)),
                    "volume_24h": float(t.get("baseVolume", 0) or t.get("quoteVolume", 0) or 0),
                    "change_pct": float(t.get("percentage", 0) or 0),
                }
            except Exception as e:
                logger.error("ccxt fetch_ticker [%s]: %s", symbol, e)
        return {"symbol": symbol, "last": 0, "volume_24h": 0, "change_pct": 0}

    def get_ohlcv(self, symbol: str, timeframe: str = "1d", limit: int = 200) -> list:
        if self._exchange:
            try:
                ohlcv = self._exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
                return ohlcv  # list of [ts, open, high, low, close, volume]
            except Exception as e:
                logger.error("ccxt fetch_ohlcv [%s]: %s", symbol, e)
        return []

    def get_balance(self, currency: str = "USDT") -> float:
        if self._exchange and self.has_keys:
            try:
                bal = self._exchange.fetch_balance()
                return float(bal.get(currency, {}).get("free", 0) if isinstance(bal.get(currency), dict) else 0)
            except Exception:
                pass
        return 0.0

    def submit_market_buy(self, symbol: str, amount: float) -> dict:
        if self._exchange and self.has_keys:
            try:
                order = self._exchange.create_market_buy_order(symbol, amount)
                return {"id": order.get("id", ""), "status": "filled", "symbol": symbol, "amount": amount}
            except Exception as e:
                return {"error": str(e)}
        # Paper mock — MUST include "filled_avg_price" key so croesus.py can read it
        ticker = self.get_ticker(symbol)
        last = ticker.get("last", 0)
        return {"id": "paper-" + symbol, "status": "filled", "symbol": symbol,
                "amount": amount, "price": last,
                "filled_avg_price": last, "paper": True}
