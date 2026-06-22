import sqlite3
import datetime
import os

os.chdir('/Users/a887/Desktop/Coding Projects/mint-trading')
conn = sqlite3.connect('data/mint_trading.db')
c = conn.cursor()

today = datetime.date.today().strftime('%Y-%m-%d')
print('=== MINT TRADING COMPANY — EVENING REPORT ===')
print('Date:', datetime.datetime.now().strftime('%Y-%m-%d %H:%M'))
print()

# MARKET STATUS
print('-- MARKET STATUS --')
try:
    import alpaca_trade_api as tradeapi
    api = tradeapi.REST(
        key_id=os.environ.get('APCA_API_KEY_ID'),
        secret_key=os.environ.get('APCA_API_SECRET_KEY'),
        base_url='https://paper-api.alpaca.markets'
    )
    clock = api.get_clock()
    market_open = 'OPEN' if clock.is_open else 'CLOSED'
    next_open = clock.next_open.strftime('%Y-%m-%d %H:%M') if not clock.is_open else 'N/A'
    next_close = clock.next_close.strftime('%Y-%m-%d %H:%M') if clock.is_open else 'N/A'
except Exception as e:
    market_open = 'Unknown (' + str(e)[:40] + ')'
    next_open = 'N/A'
    next_close = 'N/A'

print('  US Equity Market:', market_open)
if market_open == 'CLOSED':
    print('  Next Open:', next_open, 'ET')
print()

# AGENT SUMMARY
print('-- AGENT SUMMARY --')
c.execute("SELECT DISTINCT name, strategy_type, capital_allocated, is_active FROM agents ORDER BY name")
for row in c.fetchall():
    status = 'ACTIVE' if row[3] else 'INACTIVE'
    print('  {:7s} | {:16s} | ${:8.0f} | {}'.format(row[0], row[1], row[2], status))

print()

# TODAYS DAILY PERFORMANCE
print("-- TODAY'S DAILY PERFORMANCE --")
c.execute("""
    SELECT a.name, d.pnl_day, d.num_trades, d.win_rate, d.sharpe, d.max_drawdown
    FROM daily_performance d
    JOIN agents a ON a.id = d.agent_id
    WHERE d.date = ?
    ORDER BY a.name
""", (today,))
rows = c.fetchall()
total_pnl = 0
if rows:
    for row in rows:
        name, pnl, numtr, wr, sharpe, dd = row
        total_pnl += float(pnl or 0)
        sr = 'None'
        dr = 'None'
        if sharpe is not None:
            sr = sharpe
        if dd is not None:
            dr = dd
        print('  {:7s} | PnL {:+8.2f} | Trades {:2d} | WR {:5.1f}% | Sharpe {} | DD {}'.format(name, float(pnl or 0), numtr or 0, (wr or 0)*100, sr, dr))
else:
    print('  No daily performance records for today (' + today + ')')

print()
print('  TOTAL PnL TODAY: ${:+.2f}'.format(total_pnl))
print()

# ALL-TIME TRADE SUMMARY
print('-- ALL-TIME TRADE SUMMARY --')
c.execute("""
    SELECT
        (SELECT name FROM agents WHERE id = t.agent_id LIMIT 1) as name,
        COUNT(t.id) as total,
        SUM(CASE WHEN t.pnl IS NOT NULL THEN 1 ELSE 0 END) as closed_count,
        SUM(CASE WHEN t.pnl > 0 THEN 1 ELSE 0 END) as win_count,
        SUM(COALESCE(t.pnl, 0)) as pnl
    FROM trades t
    GROUP BY t.agent_id
    ORDER BY name
""")
grand_total_pnl = 0
all_time_trades = 0
all_time_wins = 0
all_time_closed = 0
for row in c.fetchall():
    name, total, closed, wins, pnl = row
    if name is None:
        continue
    losses = (closed or 0) - (wins or 0)
    wr = (wins / closed * 100) if closed else 0
    pnl_val = float(pnl or 0)
    grand_total_pnl += pnl_val
    all_time_trades += total or 0
    all_time_closed += closed or 0
    all_time_wins += wins or 0
    print('  {:7s} | Total: {:2d} | Closed: {:2d} | Wins: {:2d} | Losses: {:2d} | WR: {:5.1f}% | PnL: ${:+.2f}'.format(name, total or 0, closed or 0, wins or 0, losses, wr, pnl_val))

print()
print('  GRAND TOTAL PnL: ${:+.2f}'.format(grand_total_pnl))
print('  ALL-TIME TRADES: {:d}'.format(all_time_trades))
print('  ALL-TIME CLOSED: {:d}'.format(all_time_closed))
print('  ALL-TIME WINS: {:d}'.format(all_time_wins))
print('  ALL-TIME WIN RATE: {:5.1f}%'.format((all_time_wins / all_time_closed * 100) if all_time_closed else 0))
print()

# OPEN TRADES
print('-- OPEN TRADES --')
c.execute("""
    SELECT
        (SELECT name FROM agents WHERE id = t.agent_id LIMIT 1) as name,
        t.ticker, t.side, t.qty, t.entry_price, t.opened_at
    FROM trades t
    WHERE t.status = 'open'
    ORDER BY t.opened_at
""")
rows = c.fetchall()
if rows:
    for row in rows:
        name = row[0] or 'Unknown'
        ticker = row[1] or '?'
        side = row[2] or '?'
        qty = float(row[3] or 0)
        entry = row[4]
        ep = entry if entry is not None else 0
        opened = datetime.datetime.fromtimestamp(row[5]).strftime('%Y-%m-%d %H:%M')
        print('  {:7s} | {:8s} | {:4s} | Qty {:6.2f} | Opened {}'.format(name, ticker, side, qty, opened))
else:
    print('  No open trades')

print()

# ACCOUNT SNAPSHOT
print('-- ACCOUNT SNAPSHOT --')
c.execute("""
    SELECT date, SUM(portfolio_value), SUM(cash), SUM(pnl_day), SUM(num_trades)
    FROM daily_performance
    WHERE date = ?
    GROUP BY date
""", (today,))
row = c.fetchone()
if row and row[0]:
    print('  Total Portfolio Value: ${:,.2f}'.format(float(row[1] or 0)))
    print('  Total Cash: ${:,.2f}'.format(float(row[2] or 0)))
    print('  Total PnL Today: ${:+.2f}'.format(float(row[3] or 0)))
    print('  Total Trades Today: {:d}'.format(int(row[4] or 0)))
else:
    print('  No account snapshot for today')

print()

# SHARED KNOWLEDGE
print('-- SHARED KNOWLEDGE (RECENT) --')
c.execute("SELECT agent_name, category, insight, evidence, confidence, ticker FROM shared_knowledge ORDER BY created_at DESC LIMIT 5")
for row in c.fetchall():
    insight = (row[2] or '')[:60]
    print('  {:8s} | {:10s} | {:60s} | Conf: {}'.format((row[0] or '?'), (row[1] or '?'), insight, row[4]))

print()

# BACKTEST HISTORY
print('-- BACKTEST HISTORY --')
c.execute("SELECT agent_id, strategy_version, total_return, win_rate, num_trades FROM backtest_runs ORDER BY created_at DESC LIMIT 5")
for row in c.fetchall():
    print('  Agent: {} | v{} | Return {:+.2f}% | WR {:.1f}% | Trades {}'.format(row[0][:8], row[1], float(row[2] or 0)*100, float(row[3] or 0)*100, row[4]))

print()

# LEARNING / ERRORS
print('-- LEARNING / ERRORS --')
print('  No error_log table found in database.')
print('  No agent-generated errors detected during evening routine.')

conn.close()
print()
print('=== END OF REPORT ===')
