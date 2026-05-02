#!/usr/bin/env python3
"""Pre-launch backtest and bootstrap script — runs all 5 agents on historical data first."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.db import DatabaseManager
from core.data_pipeline import DataPipeline
from core.market import MarketInterface
from core.news_pipeline import NewsPipeline
from core.risk_manager import RiskManager
from core.crypto_market import CryptoMarket
from core.shared_knowledge import SharedKnowledge
from core.backtest_engine import BacktestEngine
from core.alternative_data import AlternativeData
from agents.athena.athena_agent import AthenaAgent
from agents.orion.orion_agent import OrionAgent
from agents.sibyl.sibyl_agent import SibylAgent
from agents.janus.janus_agent import JanusAgent
from agents.mercury.mercury_agent import MercuryAgent

db = DatabaseManager(); db.create_tables()
data = DataPipeline()
market = MarketInterface()
news = NewsPipeline(db)
risk = RiskManager(market, db)
crypto = CryptoMarket()
sk = SharedKnowledge(db)
alt = AlternativeData(data, db)
engine = BacktestEngine(data, risk, db)

# Small initial capital per agent
INITIAL = 500

# Agent definitions
configs = [
    ("Athena", lambda: AthenaAgent(INITIAL, db, market, data, news, risk), "AAPL", "MSFT", "GOOGL", "AMZN", "META", "JPM"),
    ("Orion", lambda: OrionAgent(INITIAL, db, market, data, news, risk), "NVDA", "AMD", "TSLA", "META", "NFLX", "CRM"),
    ("Sibyl", lambda: SibylAgent(INITIAL, db, market, data, news, risk), None),
    ("Janus", lambda: JanusAgent(INITIAL, db, market, data, news, risk), "SPY", "QQQ", "VIXY"),
    ("Mercury", lambda: MercuryAgent(INITIAL, db, market, data, news, risk, crypto_market=crypto, shared_knowledge=sk), "BTC/USDT", "ETH/USDT"),
]

print("=== PRE-LAUNCH BACKTEST — ALL 5 AGENTS ===\n")

for name, factory, *tickers in configs:
    try:
        agent = factory()
        t = list(tickers) if tickers else agent.UNIVERSE[:6]
        result = engine.run(agent, t, "2025-06-01", "2026-04-30", INITIAL)
        sharpe = result.get("sharpe", 0)
        total_ret = result.get("total_return", 0)
        wr = result.get("win_rate", 0)
        trades = result.get("trades", 0)
        final = result.get("final_capital", INITIAL)
        print(f"  {name:10s}  Return: {total_ret:+.1%}  Sharpe: {sharpe:+.2f}  WR: {wr:.0%}  Trades: {trades}  ${INITIAL} → ${final:,.0f}")
    except Exception as e:
        print(f"  {name:10s}  ERROR: {e}")

print("\nAll agents seeded with historical backtests. Ready for Monday.")
