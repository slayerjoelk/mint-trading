#!/usr/bin/env python3
"""Croesus — CEO of Mint Trading Company. Orchestrates all 4 agents."""

import os
import sys
import json
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.db import DatabaseManager
from core.market import MarketInterface
from core.data_pipeline import DataPipeline
from core.news_pipeline import NewsPipeline
from core.risk_manager import RiskManager
from agents.athena.athena_agent import AthenaAgent
from agents.orion.orion_agent import OrionAgent
from agents.sibyl.sibyl_agent import SibylAgent
from agents.janus.janus_agent import JanusAgent

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def _now():
    return datetime.now(timezone.utc)

def _ts():
    return int(_now().timestamp())

def _today():
    return _now().strftime("%Y-%m-%d")

def _fmt(x):
    return f"${x:+,.2f}"


db = DatabaseManager()
db.create_tables()
market = MarketInterface()
data = DataPipeline()
news = NewsPipeline(db)
risk = RiskManager(market, db)


def spawn_agents():
    agents = {
        "athena": AthenaAgent(25000, db, market, data, news, risk),
        "orion": OrionAgent(25000, db, market, data, news, risk),
        "sibyl": SibylAgent(15000, db, market, data, news, risk),
        "janus": JanusAgent(15000, db, market, data, news, risk),
    }
    print("\n=== MINT TRADING COMPANY — AGENTS SPAWNED ===\n")
    for name, agent in agents.items():
        print(f"  {name:8s}  {agent.agent_id}  ${agent.capital:,.0f}")
    print(f"\n  Total paper capital: ${sum(a.capital for a in agents.values()):,.0f}")
    print(f"  Account value: ${market.get_portfolio_value() or 0:,.0f}\n")
    return agents


def morning_routine(agents=None):
    if agents is None:
        agents = spawn_agents()
    print(f"\n=== MORNING ROUTINE — {_today()} ===\n")

    news.fetch_headlines(max_per_source=10)
    clock = market.get_clock()
    if not clock.get("is_open", True):
        print("⚠ Market closed — no orders executed.")
        return

    for name, agent in agents.items():
        print(f"\n── {name.upper()} ──")
        try:
            signals = agent.generate_signals()
        except Exception as e:
            print(f"  ✗ Signal error: {e}")
            continue

        if not signals:
            print("  (no signals)")
            continue

        for sig in signals[:5]:
            sym = sig.get("symbol", "?")
            side = sig.get("side", "buy")
            qty = sig.get("quantity", 0)
            reason = sig.get("reason", "")
            conf = sig.get("confidence", 0)

            result = risk.validate_trade(agent.agent_id, sym, qty, side, market.get_account().get("close", 0) or 0)

            if result.get("approved"):
                aqty = result.get("adjusted_quantity", qty)
                order = market.submit_order(sym, aqty, side, "market")
                agent.log_trade(sym, side, aqty, order.get("filled_avg_price", 0), reasoning=f"{reason} (conf={conf:.2f})")
                status = "✓" if "error" not in order else "✗"
                print(f"  {status} {side.upper():4s} {qty:>4d} {sym:5s}  {reason[:60]}")
            else:
                print(f"  ✗ BLOCKED {sym:5s} — {result.get('reason', '?')}")


def evening_routine(agents=None):
    if agents is None:
        agents = spawn_agents()
    print(f"\n=== EVENING ROUTINE — {_today()} ===\n")

    total_pnl = 0

    for name, agent in agents.items():
        print(f"\n── {name.upper()} ──")
        try:
            agent.learn_from_trades()
        except Exception as e:
            print(f"  ✗ Learning error: {e}")

        metrics = agent.get_performance_metrics()
        pnl = metrics.get("total_pnl", 0)
        wr = metrics.get("win_rate", 0)
        trades = metrics.get("total_trades", 0)
        total_pnl += pnl

        db.record_daily_perf(
            id=f"{agent.agent_id}-{_today()}",
            agent_id=agent.agent_id,
            date=_today(),
            portfolio_value=agent.capital,
            pnl_day=pnl,
            num_trades=trades,
            win_rate=wr,
            recorded_at=_ts(),
        )

        print(f"  PnL: {_fmt(pnl)}  |  Trades: {trades}  |  WR: {wr*100:.0f}%")

    print(f"\n  ── TOTAL PnL: {_fmt(total_pnl)} ──\n")


def weekly_review():
    print(f"\n=== WEEKLY STRATEGY REVIEW — WEEK OF {_today()} ===\n")

    agents = spawn_agents()
    rankings = []

    for name, agent in agents.items():
        m = agent.get_performance_metrics()
        rankings.append((name, m.get("sharpe_approx", 0), m.get("win_rate", 0), m.get("total_pnl", 0), m.get("total_trades", 0)))

    rankings.sort(key=lambda r: r[1], reverse=True)

    print("Rankings (by Sharpe):\n")
    for i, (name, sharpe, wr, pnl, trades) in enumerate(rankings):
        print(f"  {i+1}. {name:8s}  Sharpe={sharpe:+.2f}  WR={wr*100:.0f}%  PnL={_fmt(pnl)}  Trades={trades}")

    print("\nCapital adjustments:\n")
    for name, sharpe, wr, pnl, trades in rankings:
        if wr > 0.55 and sharpe > 1.0:
            print(f"  {name}: ↑ SCALE — increase allocation")
        elif wr < 0.40 and trades > 10:
            print(f"  {name}: ⚠ REVIEW — underperforming, reduce allocation")
        elif wr < 0.30 and trades > 20:
            print(f"  {name}: ☠ KILL — below threshold, archive learnings")
        else:
            print(f"  {name}: → HOLD — maintain current allocation")

    regime = data.get_market_regime()
    print(f"\nMarket regime: {regime}")
    print(f"Agents that benefit: {'Athena, Janus' if regime == 'ranging' else 'Orion' if regime == 'trending_up' else 'Janus (defensive)' if regime == 'trending_down' else 'All'}")

    report_path = os.path.join(ROOT, ".hermes", "weekly-reports", f"{_today()}.md")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        f.write(f"# Weekly Review — {_today()}\n\n")
        f.write("## Agent Rankings\n\n")
        for i, (name, sharpe, wr, pnl, trades) in enumerate(rankings):
            f.write(f"- **{i+1}. {name}**: Sharpe={sharpe:+.2f}, WR={wr*100:.0f}%, PnL={_fmt(pnl)}, Trades={trades}\n")
        f.write(f"\n## Market Regime: {regime}\n")
    print(f"\nReport saved: {report_path}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "spawn"
    if cmd == "morning":
        morning_routine()
    elif cmd == "evening":
        evening_routine()
    elif cmd == "weekly":
        weekly_review()
    else:
        spawn_agents()
