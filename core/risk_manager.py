from datetime import datetime, timezone, date


def _today_unix_range() -> tuple[int, int]:
    """Return Unix timestamps for start and end of today (UTC)."""
    today = datetime.now(timezone.utc).date()
    start = int(datetime(today.year, today.month, today.day, tzinfo=timezone.utc).timestamp())
    end = start + 86400
    return start, end


class RiskManager:
    def __init__(
        self,
        market,
        db,
        max_position_pct: float = 0.02,
        max_portfolio_exposure: float = 0.80,
        max_daily_trades: int = 5,
        max_drawdown_pct: float = 0.25,
    ):
        self.market = market
        self.db = db
        self.max_position_pct = max_position_pct
        self.max_portfolio_exposure = max_portfolio_exposure
        self.max_daily_trades = max_daily_trades
        self.max_drawdown_pct = max_drawdown_pct

    def validate_trade(
        self, agent_id: str, symbol: str, quantity: int, side: str, price: float
    ) -> dict:
        # Market hours check
        clock = self.market.get_clock()
        if isinstance(clock, dict) and "error" not in clock and not clock.get("is_open", True):
            return {"approved": False, "reason": "Market is closed", "adjusted_quantity": 0}

        account = self.market.get_account()
        if "error" in account:
            return {
                "approved": False,
                "reason": f"Cannot fetch account: {account['error']}",
                "adjusted_quantity": 0,
            }

        portfolio_value = float(account.get("portfolio_value") or 0)
        cash = float(account.get("cash") or 0)

        if portfolio_value <= 0:
            return {"approved": False, "reason": "Portfolio value is zero", "adjusted_quantity": 0}

        # Daily trade count
        limits = self.get_trade_limits(agent_id)
        if limits["trades_remaining"] <= 0:
            return {"approved": False, "reason": "Daily trade limit reached", "adjusted_quantity": 0}

        # Drawdown gate
        if limits["in_drawdown"]:
            return {
                "approved": False,
                "reason": "Agent in max drawdown — trading suspended",
                "adjusted_quantity": 0,
            }

        # Position size limit
        max_value = portfolio_value * self.max_position_pct
        adjusted_qty = quantity

        if adjusted_qty * price > max_value:
            adjusted_qty = int(max_value / price)
        if adjusted_qty <= 0:
            return {
                "approved": False,
                "reason": "Requested position exceeds max size and minimum is below 1 share",
                "adjusted_quantity": 0,
            }

        # Cash check for buys
        if side == "buy":
            if adjusted_qty * price > cash:
                adjusted_qty = int(cash / price)
            if adjusted_qty <= 0:
                return {"approved": False, "reason": "Insufficient cash", "adjusted_quantity": 0}

        # Portfolio exposure check (buys only)
        if side == "buy" and limits["current_exposure_pct"] >= self.max_portfolio_exposure:
            return {
                "approved": False,
                "reason": f"Portfolio exposure {limits['current_exposure_pct']:.1%} at max ({self.max_portfolio_exposure:.0%})",
                "adjusted_quantity": 0,
            }

        reason = "Approved" if adjusted_qty == quantity else f"Approved — quantity reduced {quantity}→{adjusted_qty}"
        return {"approved": True, "reason": reason, "adjusted_quantity": adjusted_qty}

    def calculate_position_size(
        self, agent_capital: float, price: float, risk_per_trade: float = 0.02
    ) -> int:
        if price <= 0 or agent_capital <= 0:
            return 0
        stop_distance = price * 0.02  # 2% of price as default stop
        risk_amount = agent_capital * risk_per_trade
        shares = int(risk_amount / stop_distance)
        max_shares = int((agent_capital * self.max_position_pct) / price)
        return max(0, min(shares, max_shares))

    def calculate_position_size_atr(
        self, agent_capital: float, price: float, atr: float, risk_per_trade: float = 0.02
    ) -> int:
        """Size position using 1.5× ATR as stop distance."""
        if price <= 0 or agent_capital <= 0 or atr <= 0:
            return self.calculate_position_size(agent_capital, price, risk_per_trade)
        stop_distance = atr * 1.5
        risk_amount = agent_capital * risk_per_trade
        shares = int(risk_amount / stop_distance)
        max_shares = int((agent_capital * self.max_position_pct) / price)
        return max(0, min(shares, max_shares))

    def check_portfolio_exposure(self, agent_id: str) -> float:
        account = self.market.get_account()
        if "error" in account:
            return 0.0
        portfolio_value = float(account.get("portfolio_value") or 0)
        if portfolio_value <= 0:
            return 0.0
        positions = self.market.get_positions()
        if not positions or isinstance(positions, dict):
            return 0.0
        total_equity = sum(float(p.get("market_value") or 0) for p in positions)
        return total_equity / portfolio_value

    def is_agent_in_drawdown(self, agent_id: str) -> bool:
        # Pull from daily_performance if available, otherwise use live account
        try:
            conn = self.db._conn()
            rows = conn.execute(
                """
                SELECT portfolio_value FROM daily_performance
                WHERE agent_id = ? ORDER BY date DESC LIMIT 30
                """,
                (agent_id,),
            ).fetchall()
            conn.close()
            if rows:
                values = [float(r["portfolio_value"] or 0) for r in rows]
                peak = max(values)
                current = values[0]
                if peak > 0 and (peak - current) / peak >= self.max_drawdown_pct:
                    return True
                return False
        except Exception:
            pass

        # Fallback: compare portfolio_value vs equity as rough proxy
        account = self.market.get_account()
        if "error" in account:
            return False
        portfolio_value = float(account.get("portfolio_value") or 0)
        equity = float(account.get("equity") or 0)
        if equity > 0 and portfolio_value < equity * (1 - self.max_drawdown_pct):
            return True
        return False

    def get_trade_limits(self, agent_id: str) -> dict:
        today_start, today_end = _today_unix_range()
        trades_today = 0
        try:
            conn = self.db._conn()
            row = conn.execute(
                "SELECT COUNT(*) AS cnt FROM trades WHERE agent_id = ? AND opened_at >= ? AND opened_at < ?",
                (agent_id, today_start, today_end),
            ).fetchone()
            conn.close()
            trades_today = row["cnt"] if row else 0
        except Exception:
            pass

        exposure = self.check_portfolio_exposure(agent_id)
        return {
            "trades_today": trades_today,
            "trades_remaining": max(0, self.max_daily_trades - trades_today),
            "max_daily_trades": self.max_daily_trades,
            "current_exposure_pct": round(exposure, 4),
            "max_exposure_pct": self.max_portfolio_exposure,
            "in_drawdown": self.is_agent_in_drawdown(agent_id),
        }

    def kill_switch(self, agent_id: str) -> dict:
        """Liquidate all positions and deactivate the agent."""
        positions = self.market.get_positions()
        liquidations = []
        if positions and not isinstance(positions, dict):
            for pos in positions:
                symbol = pos.get("symbol")
                qty = abs(int(float(pos.get("qty") or 0)))
                side = pos.get("side", "long")
                order_side = "sell" if side == "long" else "buy"
                if qty > 0:
                    result = self.market.submit_order(symbol, qty, order_side, "market")
                    liquidations.append({"symbol": symbol, "qty": qty, "result": result})

        try:
            conn = self.db._conn()
            conn.execute("UPDATE agents SET is_active = 0 WHERE id = ?", (agent_id,))
            conn.commit()
            conn.close()
        except Exception as exc:
            print(f"[RiskManager] kill_switch DB error: {exc}")

        print(f"[RiskManager] KILL SWITCH fired for agent {agent_id}. {len(liquidations)} positions liquidated.")
        return {"agent_id": agent_id, "status": "killed", "liquidations": liquidations}
