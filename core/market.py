import os
import logging
from typing import Optional

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest, StopLimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderType
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

logger = logging.getLogger(__name__)

PAPER = True


class MarketInterface:
    def __init__(self, api_key: str = None, secret_key: str = None):
        self.api_key = api_key or os.environ.get("APCA_API_KEY_ID", "")
        self.secret_key = secret_key or os.environ.get("APCA_API_SECRET_KEY", "")
        self._trading = TradingClient(self.api_key, self.secret_key, paper=PAPER)
        self._data = StockHistoricalDataClient(self.api_key, self.secret_key)

    # ── Account ──────────────────────────────────────────────────────────────

    def get_account(self) -> dict:
        try:
            acct = self._trading.get_account()
            return {
                "id": str(acct.id),
                "status": str(acct.status),
                "cash": float(acct.cash),
                "portfolio_value": float(acct.portfolio_value),
                "equity": float(acct.equity),
                "buying_power": float(acct.buying_power),
                "daytrade_count": int(acct.daytrade_count),
                "pattern_day_trader": bool(acct.pattern_day_trader),
                "trading_blocked": bool(acct.trading_blocked),
                "currency": str(acct.currency),
            }
        except Exception as e:
            logger.error("get_account error: %s", e)
            return {"error": str(e)}

    def get_positions(self) -> list[dict]:
        try:
            positions = self._trading.get_all_positions()
            return [
                {
                    "symbol": p.symbol,
                    "qty": float(p.qty),
                    "side": str(p.side),
                    "avg_entry_price": float(p.avg_entry_price),
                    "current_price": float(p.current_price),
                    "market_value": float(p.market_value),
                    "unrealized_pl": float(p.unrealized_pl),
                    "unrealized_plpc": float(p.unrealized_plpc),
                }
                for p in positions
            ]
        except Exception as e:
            logger.error("get_positions error: %s", e)
            return []

    def get_portfolio_value(self) -> Optional[float]:
        acct = self.get_account()
        if "error" in acct:
            return None
        return acct["portfolio_value"]

    # ── Orders ────────────────────────────────────────────────────────────────

    def submit_order(self, symbol: str, qty: float, side: str, type: str,
                     limit_price: float = None, stop_price: float = None) -> dict:
        try:
            _side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
            _type = type.lower()
            # Crypto symbols contain "/" (e.g. BTC/USD). Alpaca crypto only accepts GTC or IOC, not DAY.
            is_crypto = "/" in symbol
            tif = TimeInForce.GTC if is_crypto else TimeInForce.DAY

            if _type == "market":
                req = MarketOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=_side,
                    time_in_force=tif,
                )
            elif _type == "limit":
                if limit_price is None:
                    return {"error": "limit_price required for limit orders"}
                req = LimitOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=_side,
                    time_in_force=tif,
                    limit_price=limit_price,
                )
            elif _type == "stop_limit":
                if limit_price is None or stop_price is None:
                    return {"error": "limit_price and stop_price required for stop_limit orders"}
                req = StopLimitOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=_side,
                    time_in_force=tif,
                    limit_price=limit_price,
                    stop_price=stop_price,
                )
            else:
                return {"error": f"Unknown order type: {type}"}

            order = self._trading.submit_order(req)
            return self._order_to_dict(order)
        except Exception as e:
            logger.error("submit_order error [%s %s %s]: %s", side, qty, symbol, e)
            return {"error": str(e)}

    def cancel_all_orders(self) -> dict:
        try:
            self._trading.cancel_orders()
            return {"status": "ok"}
        except Exception as e:
            logger.error("cancel_all_orders error: %s", e)
            return {"error": str(e)}

    def get_order_status(self, order_id: str) -> dict:
        try:
            order = self._trading.get_order_by_id(order_id)
            return self._order_to_dict(order)
        except Exception as e:
            logger.error("get_order_status error [%s]: %s", order_id, e)
            return {"error": str(e)}

    # ── Market Data ───────────────────────────────────────────────────────────

    def get_bars(self, symbols: list[str], timeframe: str = "1Day", limit: int = 100) -> dict:
        try:
            _tf_map = {
                "1Min": TimeFrame(1, TimeFrameUnit.Minute),
                "5Min": TimeFrame(5, TimeFrameUnit.Minute),
                "15Min": TimeFrame(15, TimeFrameUnit.Minute),
                "1Hour": TimeFrame(1, TimeFrameUnit.Hour),
                "1Day": TimeFrame(1, TimeFrameUnit.Day),
            }
            tf = _tf_map.get(timeframe, TimeFrame(1, TimeFrameUnit.Day))

            req = StockBarsRequest(symbol_or_symbols=symbols, timeframe=tf, limit=limit)
            bars = self._data.get_stock_bars(req)

            result = {}
            for sym, bar_list in bars.data.items():
                result[sym] = [
                    {
                        "timestamp": str(b.timestamp),
                        "open": float(b.open),
                        "high": float(b.high),
                        "low": float(b.low),
                        "close": float(b.close),
                        "volume": float(b.volume),
                        "vwap": float(b.vwap) if b.vwap else None,
                    }
                    for b in bar_list
                ]
            return result
        except Exception as e:
            logger.error("get_bars error: %s", e)
            return {"error": str(e)}

    def get_clock(self) -> dict:
        try:
            clock = self._trading.get_clock()
            return {
                "is_open": bool(clock.is_open),
                "timestamp": str(clock.timestamp),
                "next_open": str(clock.next_open),
                "next_close": str(clock.next_close),
            }
        except Exception as e:
            logger.error("get_clock error: %s", e)
            return {"error": str(e)}

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _order_to_dict(self, order) -> dict:
        return {
            "id": str(order.id),
            "client_order_id": str(order.client_order_id),
            "symbol": str(order.symbol),
            "qty": float(order.qty) if order.qty else None,
            "filled_qty": float(order.filled_qty) if order.filled_qty else 0.0,
            "side": str(order.side),
            "type": str(order.order_type),
            "status": str(order.status),
            "limit_price": float(order.limit_price) if order.limit_price else None,
            "stop_price": float(order.stop_price) if order.stop_price else None,
            "filled_avg_price": float(order.filled_avg_price) if order.filled_avg_price else None,
            "created_at": str(order.created_at),
            "updated_at": str(order.updated_at),
        }
