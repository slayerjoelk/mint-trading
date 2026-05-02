import uuid
import statistics
from abc import abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import yaml
from scipy import stats  # for statistical tests (shapiro, ttest, skew, kurtosis)

_AGENTS_ROOT = Path(__file__).parent


def _now() -> int:
    return int(datetime.now(timezone.utc).timestamp())


class BaseAgent:
    def __init__(
        self,
        name: str,
        style: str,
        asset_focus: str,
        initial_capital: float,
        db,
        market,
        data,
        news,
        risk,
    ):
        self.agent_id = str(uuid.uuid4())
        self.name = name
        self.style = style
        self.asset_focus = asset_focus
        self._initial_capital = initial_capital
        self.db = db
        self.market = market
        self.data = data
        self.news = news
        self.risk = risk

        self._agent_dir = _AGENTS_ROOT / name.lower().replace(" ", "_")
        self._agent_dir.mkdir(parents=True, exist_ok=True)

        self._config_path = self._agent_dir / "config.yaml"
        self._config = self._load_or_create_config()

        db.insert_agent(
            id=self.agent_id,
            name=name,
            strategy_type=style,
            risk_tolerance=0.5,
            capital_allocated=initial_capital,
            created_at=_now(),
            meta={"asset_focus": asset_focus},
        )

    # ------------------------------------------------------------------ config

    def _load_or_create_config(self) -> dict:
        if self._config_path.exists():
            with open(self._config_path) as f:
                return yaml.safe_load(f) or {}
        defaults = {
            "name": self.name,
            "style": self.style,
            "asset_focus": self.asset_focus,
            "initial_capital": self._initial_capital,
            "risk_per_trade": 0.02,
            "max_positions": 5,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        with open(self._config_path, "w") as f:
            yaml.dump(defaults, f, default_flow_style=False)
        return defaults

    def get_config(self) -> dict:
        return dict(self._config)

    def save_config(self, updates: Optional[dict] = None) -> None:
        if updates:
            self._config.update(updates)
        with open(self._config_path, "w") as f:
            yaml.dump(self._config, f, default_flow_style=False)

    # ---------------------------------------------------------------- properties

    @property
    def capital(self) -> float:
        """Agent's tracked capital (initial + closed P&L from DB)."""
        metrics = self.get_performance_metrics()
        return self._initial_capital + metrics.get("total_pnl", 0.0)

    @property
    def portfolio_value(self) -> float:
        """Full account value from Alpaca — all agents share this."""
        account = self.market.get_account()
        if "error" in account:
            return self._initial_capital
        return float(account.get("portfolio_value") or self._initial_capital)

    @property
    def positions(self) -> list:
        result = self.market.get_positions()
        if isinstance(result, dict) and "error" in result:
            return []
        return result or []

    @property
    def pnl(self) -> float:
        metrics = self.get_performance_metrics()
        return metrics.get("total_pnl", 0.0)

    @property
    def win_rate(self) -> float:
        return self.get_performance_metrics().get("win_rate", 0.0)

    @property
    def total_trades(self) -> int:
        return self.get_performance_metrics().get("total_trades", 0)

    @property
    def sharpe_approx(self) -> float:
        return self.get_performance_metrics().get("sharpe_approx", 0.0)

    # ------------------------------------------------- abstract interface

    @abstractmethod
    def research(self):
        """Run a research sprint. Produce a thesis doc and strategy file."""

    @abstractmethod
    def generate_signals(self) -> list[dict]:
        """
        Return signals: [{symbol, side, quantity, type, reason, confidence}, ...]
        side: 'buy' | 'sell'
        type: 'market' | 'limit'
        confidence: 0.0–1.0
        """

    @abstractmethod
    def learn_from_trades(self):
        """Review yesterday's closed trades. Update thesis and save new strategy version."""

    # ------------------------------------------------- shared methods

    def log_trade(
        self,
        symbol: str,
        side: str,
        qty: int,
        entry: float,
        exit_price: Optional[float] = None,
        pnl: Optional[float] = None,
        reasoning: str = "",
    ) -> str:
        trade_id = str(uuid.uuid4())
        now = _now()
        status = "closed" if exit_price is not None else "open"

        self.db.log_trade(
            id=trade_id,
            agent_id=self.agent_id,
            ticker=symbol,
            side=side,
            qty=qty,
            entry_price=entry,
            exit_price=exit_price,
            pnl=pnl,
            status=status,
            opened_at=now,
            closed_at=now if status == "closed" else None,
        )

        if status == "closed":
            self.journal_entry(
                trade_id=trade_id,
                analysis=reasoning,
                lessons=None,
                symbol=symbol,
                entry=entry,
                exit_price=exit_price,
                pnl=pnl or 0.0,
            )

        return trade_id

    def get_performance_metrics(self) -> dict:
        _empty = {
            "win_rate": 0.0, "total_trades": 0, "total_pnl": 0.0,
            "avg_win": 0.0, "avg_loss": 0.0, "profit_factor": 0.0,
            "sharpe_approx": 0.0, "max_drawdown": 0.0, "current_drawdown": 0.0,
        }
        try:
            conn = self.db._conn()
            rows = conn.execute(
                "SELECT pnl FROM trades WHERE agent_id = ? AND status = 'closed'",
                (self.agent_id,),
            ).fetchall()
            conn.close()
        except Exception:
            return _empty

        if not rows:
            return _empty

        pnls = [float(r["pnl"] or 0) for r in rows]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        total = len(pnls)

        gross_win = sum(wins)
        gross_loss = abs(sum(losses))

        if len(pnls) > 1:
            mean_p = statistics.mean(pnls)
            std_p = statistics.stdev(pnls)
            sharpe = (mean_p / std_p * (252 ** 0.5)) if std_p > 0 else 0.0
        else:
            sharpe = 0.0

        # Walk-forward drawdown
        cumulative, peak, max_dd = 0.0, 0.0, 0.0
        for p in pnls:
            cumulative += p
            peak = max(peak, cumulative)
            dd = (peak - cumulative) / peak if peak > 0 else 0.0
            max_dd = max(max_dd, dd)
        current_dd = (peak - cumulative) / peak if peak > 0 else 0.0

        return {
            "win_rate": round(len(wins) / total, 4),
            "total_trades": total,
            "total_pnl": round(sum(pnls), 2),
            "avg_win": round(gross_win / len(wins), 2) if wins else 0.0,
            "avg_loss": round(-gross_loss / len(losses), 2) if losses else 0.0,
            "profit_factor": round(gross_win / gross_loss, 4) if gross_loss > 0 else 0.0,
            "sharpe_approx": round(sharpe, 4),
            "max_drawdown": round(max_dd, 4),
            "current_drawdown": round(current_dd, 4),
        }

    def save_strategy(self, version: str, description: str, code_str: str) -> str:
        strategy_id = str(uuid.uuid4())
        path = self._agent_dir / f"strategy_v{version}.py"
        path.write_text(code_str)
        self.db.save_strategy(
            id=strategy_id,
            agent_id=self.agent_id,
            name=f"{self.name}_v{version}",
            description=description,
            parameters={"version": version, "path": str(path)},
            version=version,
            created_at=_now(),
        )
        return strategy_id

    def load_strategy(self, version: str) -> Optional[str]:
        path = self._agent_dir / f"strategy_v{version}.py"
        return path.read_text() if path.exists() else None

    def journal_entry(
        self,
        trade_id: str,
        analysis: str,
        lessons: Optional[str],
        symbol: Optional[str] = None,
        entry: Optional[float] = None,
        exit_price: Optional[float] = None,
        pnl: float = 0.0,
    ) -> None:
        journal_id = str(uuid.uuid4())

        market_conditions: dict = {}
        if symbol:
            try:
                features = self.data.get_features(symbol)
                regime = self.data.get_market_regime()
                market_conditions = {
                    "regime": regime,
                    "rsi_14": features.get("rsi_14"),
                    "macd_hist": features.get("macd_hist"),
                    "vol_ratio": features.get("vol_ratio"),
                    "atr_14": features.get("atr_14"),
                    "price_vs_sma20": features.get("price_vs_sma20"),
                    "close": features.get("close"),
                }
            except Exception:
                pass

        outcome = "WIN" if pnl > 0 else ("LOSS" if pnl < 0 else "FLAT")
        regime = market_conditions.get("regime", "unknown")
        rsi = market_conditions.get("rsi_14", "N/A")
        macd_hist = market_conditions.get("macd_hist", "N/A")
        vol_ratio = market_conditions.get("vol_ratio", "N/A")
        pnl_pct_str = ""
        if entry and entry > 0 and exit_price:
            pnl_pct = ((exit_price - entry) / entry) * 100
            if analysis and "sell" in analysis.lower():
                pnl_pct = -pnl_pct
            pnl_pct_str = f" ({pnl_pct:+.2f}%)"

        entry_str = f"${entry:.4f}" if entry else "N/A"
        exit_str = f"${exit_price:.4f}" if exit_price else "N/A"

        post_analysis = (
            f"[{outcome}] {symbol or 'UNKNOWN'} — PnL: ${pnl:+.2f}{pnl_pct_str}\n"
            f"Entry: {entry_str}  |  Exit: {exit_str}\n"
            f"Market regime at close: {regime}\n"
            f"Indicators: RSI={rsi} | MACD_hist={macd_hist} | Vol_ratio={vol_ratio}\n"
            f"\nWhat I expected:\n{analysis}\n"
            f"\nWhat actually happened:\n"
            f"The trade {'hit its target' if pnl > 0 else 'did not reach its target' if pnl < 0 else 'closed flat'}. "
            f"The regime was '{regime}', which {'supported' if pnl > 0 else 'did not support'} the thesis.\n"
        )

        if lessons:
            learned = lessons
        elif pnl > 0:
            learned = (
                f"Setup worked. Confirm signal confluence that produced this {outcome}: "
                f"RSI={rsi}, MACD hist={macd_hist}, vol_ratio={vol_ratio}. "
                "Flag whether this regime repeats and if the same entry criteria should be prioritised."
            )
        else:
            learned = (
                f"Review entry criteria. RSI={rsi}, MACD hist={macd_hist}, vol_ratio={vol_ratio} "
                f"in a '{regime}' market did not produce the expected move. "
                "Consider: (1) Was the stop too tight relative to ATR? "
                "(2) Did regime contradict the signal? "
                "(3) Was the position sized correctly? "
                "Update strategy parameters before next trade."
            )

        self.db.log_journal_entry(
            id=journal_id,
            trade_id=trade_id,
            agent_id=self.agent_id,
            ticker=symbol or "",
            entry_price=entry,
            exit_price=exit_price,
            pnl=pnl,
            pre_trade_reasoning=analysis,
            post_trade_analysis=post_analysis,
            market_conditions=market_conditions,
            lessons_learned=learned,
            timestamp=_now(),
        )
