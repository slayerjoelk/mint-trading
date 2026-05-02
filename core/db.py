import sqlite3
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent.parent / "data" / "mint_trading.db"


class DatabaseManager:
    def __init__(self, db_path: str = None):
        self.db_path = str(db_path or DB_PATH)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def create_tables(self):
        ddl = """
        CREATE TABLE IF NOT EXISTS agents (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            strategy_type TEXT,
            risk_tolerance REAL,
            capital_allocated REAL,
            is_active INTEGER DEFAULT 1,
            created_at INTEGER,
            meta TEXT
        );

        CREATE TABLE IF NOT EXISTS strategies (
            id TEXT PRIMARY KEY,
            agent_id TEXT,
            name TEXT NOT NULL,
            description TEXT,
            parameters TEXT,
            version TEXT,
            created_at INTEGER,
            FOREIGN KEY (agent_id) REFERENCES agents(id)
        );

        CREATE TABLE IF NOT EXISTS trades (
            id TEXT PRIMARY KEY,
            agent_id TEXT,
            ticker TEXT NOT NULL,
            side TEXT NOT NULL,
            qty REAL NOT NULL,
            entry_price REAL,
            exit_price REAL,
            pnl REAL,
            status TEXT DEFAULT 'open',
            order_id TEXT,
            strategy_id TEXT,
            opened_at INTEGER,
            closed_at INTEGER,
            FOREIGN KEY (agent_id) REFERENCES agents(id)
        );

        CREATE TABLE IF NOT EXISTS trade_journal (
            id TEXT PRIMARY KEY,
            trade_id TEXT NOT NULL,
            agent_id TEXT NOT NULL,
            ticker TEXT NOT NULL,
            entry_price REAL,
            exit_price REAL,
            pnl REAL,
            pre_trade_reasoning TEXT,
            post_trade_analysis TEXT,
            market_conditions TEXT,
            lessons_learned TEXT,
            timestamp INTEGER,
            FOREIGN KEY (trade_id) REFERENCES trades(id),
            FOREIGN KEY (agent_id) REFERENCES agents(id)
        );

        CREATE TABLE IF NOT EXISTS signals (
            id TEXT PRIMARY KEY,
            agent_id TEXT,
            ticker TEXT NOT NULL,
            signal_type TEXT NOT NULL,
            direction TEXT,
            strength REAL,
            source TEXT,
            payload TEXT,
            created_at INTEGER,
            FOREIGN KEY (agent_id) REFERENCES agents(id)
        );

        CREATE TABLE IF NOT EXISTS news_events (
            id TEXT PRIMARY KEY,
            ticker TEXT,
            headline TEXT,
            summary TEXT,
            sentiment REAL,
            source TEXT,
            published_at INTEGER,
            recorded_at INTEGER
        );

        CREATE TABLE IF NOT EXISTS daily_performance (
            id TEXT PRIMARY KEY,
            agent_id TEXT,
            date TEXT NOT NULL,
            portfolio_value REAL,
            cash REAL,
            equity REAL,
            pnl_day REAL,
            pnl_total REAL,
            num_trades INTEGER,
            win_rate REAL,
            sharpe REAL,
            max_drawdown REAL,
            recorded_at INTEGER,
            FOREIGN KEY (agent_id) REFERENCES agents(id)
        );

        CREATE TABLE IF NOT EXISTS shared_knowledge (
            id TEXT PRIMARY KEY,
            agent_id TEXT NOT NULL,
            agent_name TEXT NOT NULL,
            category TEXT NOT NULL,
            insight TEXT NOT NULL,
            evidence TEXT,
            confidence REAL NOT NULL,
            ticker TEXT,
            created_at INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS backtest_runs (
            id TEXT PRIMARY KEY,
            agent_id TEXT NOT NULL,
            strategy_version TEXT,
            start_date TEXT NOT NULL,
            end_date TEXT NOT NULL,
            initial_capital REAL NOT NULL,
            final_capital REAL NOT NULL,
            total_return REAL NOT NULL,
            sharpe REAL,
            max_drawdown REAL,
            win_rate REAL,
            num_trades INTEGER,
            daily_returns TEXT,
            created_at INTEGER NOT NULL
        );
        """
        with self._conn() as conn:
            conn.executescript(ddl)
        logger.info("Tables created/verified at %s", self.db_path)

    def insert_agent(self, id: str, name: str, strategy_type: str = None,
                     risk_tolerance: float = None, capital_allocated: float = None,
                     created_at: int = None, meta: dict = None):
        sql = """
        INSERT OR REPLACE INTO agents
            (id, name, strategy_type, risk_tolerance, capital_allocated, created_at, meta)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        with self._conn() as conn:
            conn.execute(sql, (
                id, name, strategy_type, risk_tolerance, capital_allocated,
                created_at, json.dumps(meta) if meta else None
            ))

    def log_trade(self, id: str, agent_id: str, ticker: str, side: str, qty: float,
                  entry_price: float = None, exit_price: float = None, pnl: float = None,
                  status: str = "open", order_id: str = None, strategy_id: str = None,
                  opened_at: int = None, closed_at: int = None):
        sql = """
        INSERT OR REPLACE INTO trades
            (id, agent_id, ticker, side, qty, entry_price, exit_price, pnl,
             status, order_id, strategy_id, opened_at, closed_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        with self._conn() as conn:
            conn.execute(sql, (
                id, agent_id, ticker, side, qty, entry_price, exit_price, pnl,
                status, order_id, strategy_id, opened_at, closed_at
            ))

    def log_journal_entry(self, id: str, trade_id: str, agent_id: str, ticker: str,
                          entry_price: float = None, exit_price: float = None,
                          pnl: float = None, pre_trade_reasoning: str = None,
                          post_trade_analysis: str = None, market_conditions: dict = None,
                          lessons_learned: str = None, timestamp: int = None):
        sql = """
        INSERT OR REPLACE INTO trade_journal
            (id, trade_id, agent_id, ticker, entry_price, exit_price, pnl,
             pre_trade_reasoning, post_trade_analysis, market_conditions,
             lessons_learned, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        with self._conn() as conn:
            conn.execute(sql, (
                id, trade_id, agent_id, ticker, entry_price, exit_price, pnl,
                pre_trade_reasoning, post_trade_analysis,
                json.dumps(market_conditions) if market_conditions else None,
                lessons_learned, timestamp
            ))

    def save_strategy(self, id: str, agent_id: str, name: str, description: str = None,
                      parameters: dict = None, version: str = "1.0", created_at: int = None):
        sql = """
        INSERT OR REPLACE INTO strategies
            (id, agent_id, name, description, parameters, version, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        with self._conn() as conn:
            conn.execute(sql, (
                id, agent_id, name, description,
                json.dumps(parameters) if parameters else None,
                version, created_at
            ))

    def record_news(self, id: str, ticker: str = None, headline: str = None,
                    summary: str = None, sentiment: float = None, source: str = None,
                    published_at: int = None, recorded_at: int = None):
        sql = """
        INSERT OR REPLACE INTO news_events
            (id, ticker, headline, summary, sentiment, source, published_at, recorded_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        with self._conn() as conn:
            conn.execute(sql, (
                id, ticker, headline, summary, sentiment, source, published_at, recorded_at
            ))

    def record_signal(self, id: str, agent_id: str = None, ticker: str = None,
                      signal_type: str = None, direction: str = None,
                      strength: float = None, source: str = None,
                      payload: dict = None, created_at: int = None):
        sql = """
        INSERT OR REPLACE INTO signals
            (id, agent_id, ticker, signal_type, direction, strength, source, payload, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        with self._conn() as conn:
            conn.execute(sql, (
                id, agent_id, ticker, signal_type, direction, strength, source,
                json.dumps(payload) if payload else None, created_at
            ))

    def insert_shared_knowledge(self, id: str, agent_id: str, agent_name: str,
                                category: str, insight: str, evidence: dict = None,
                                confidence: float = 0.5, ticker: str = None,
                                created_at: int = None):
        sql = """
        INSERT OR REPLACE INTO shared_knowledge
            (id, agent_id, agent_name, category, insight, evidence, confidence, ticker, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        with self._conn() as conn:
            conn.execute(sql, (
                id, agent_id, agent_name, category, insight,
                json.dumps(evidence) if evidence else None,
                confidence, ticker, created_at
            ))

    def insert_backtest_run(self, id: str, agent_id: str, strategy_version: str,
                            start_date: str, end_date: str, initial_capital: float,
                            final_capital: float, total_return: float, sharpe: float = None,
                            max_drawdown: float = None, win_rate: float = None,
                            num_trades: int = None, daily_returns: list = None,
                            created_at: int = None):
        sql = """
        INSERT OR REPLACE INTO backtest_runs
            (id, agent_id, strategy_version, start_date, end_date, initial_capital,
             final_capital, total_return, sharpe, max_drawdown, win_rate, num_trades,
             daily_returns, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        with self._conn() as conn:
            conn.execute(sql, (
                id, agent_id, strategy_version, start_date, end_date, initial_capital,
                final_capital, total_return, sharpe, max_drawdown, win_rate, num_trades,
                json.dumps(daily_returns) if daily_returns is not None else None,
                created_at
            ))

    def record_daily_perf(self, id: str, agent_id: str, date: str,
                          portfolio_value: float = None, cash: float = None,
                          equity: float = None, pnl_day: float = None,
                          pnl_total: float = None, num_trades: int = None,
                          win_rate: float = None, sharpe: float = None,
                          max_drawdown: float = None, recorded_at: int = None):
        sql = """
        INSERT OR REPLACE INTO daily_performance
            (id, agent_id, date, portfolio_value, cash, equity, pnl_day, pnl_total,
             num_trades, win_rate, sharpe, max_drawdown, recorded_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        with self._conn() as conn:
            conn.execute(sql, (
                id, agent_id, date, portfolio_value, cash, equity, pnl_day, pnl_total,
                num_trades, win_rate, sharpe, max_drawdown, recorded_at
            ))
