import sqlite3, os

db_path = os.path.expanduser('~/Desktop/Coding Projects/mint-trading/data/mint_trading.db')
conn = sqlite3.connect(db_path)
c = conn.cursor()

print('--- Mint Trading DB ---')

c.execute("SELECT COUNT(*) FROM agents")
print(f"Agents rows: {c.fetchone()[0]}")

c.execute("SELECT COUNT(DISTINCT id) FROM agents")
print(f"Distinct agent IDs: {c.fetchone()[0]}")

c.execute("SELECT COUNT(*), status FROM trades GROUP BY status")
print(f"Trades by status: {c.fetchall()}")

c.execute("SELECT name, id FROM agents")
print(f"Agents:")
for n, i in c.fetchall():
    print(f"  {n} ({i[:8]}...)")

c.execute("""
    SELECT 
        a.name,
        COALESCE(SUM(t.pnl), 0) AS total,
        COUNT(*) AS cnt,
        COALESCE(SUM(CASE WHEN t.pnl > 0 THEN 1 ELSE 0 END) * 100.0 / NULLIF(COUNT(*), 0), 0) AS wr
    FROM agents a
    JOIN trades t ON a.id = t.agent_id
    GROUP BY a.name
""")
print("\nPer-agent (DB JOIN):")
for row in c.fetchall():
    print(f"  {row[0]}: PnL ${row[1]:+.2f} | Trades {row[2]} | WR {row[3]:.0f}%")

# Total PnL
c.execute("SELECT COALESCE(SUM(pnl), 0) FROM trades WHERE status = 'closed'")
print(f"\nTotal closed PnL: ${c.fetchone()[0]:+.2f}")

# Open trades
c.execute("""
    SELECT t.ticker, a.name, t.entry_price, t.pnl, t.status, t.opened_at
    FROM trades t
    JOIN agents a ON t.agent_id = a.id
    WHERE t.status = 'open'
""")
open_trades = c.fetchall()
print(f"\nOpen trades: {len(open_trades)}")
for t in open_trades:
    print(f"  {t[0]} ({t[1]}): entry=${t[2]}, pnl={t[3]}, status={t[4]}, opened_at={t[5]}")

# Daily performance (today)
c.execute("""
    SELECT a.name, dp.pnl_day, dp.num_trades, dp.win_rate, dp.portfolio_value
    FROM daily_performance dp
    JOIN agents a ON dp.agent_id = a.id
    WHERE dp.date = (SELECT MAX(date) FROM daily_performance)
    ORDER BY dp.pnl_day DESC
""")
rows = c.fetchall()
if rows:
    print(f"\nDaily performance for {rows[0][4]}:")
    for r in rows:
        print(f"  {r[0]}: PnL_day=${r[1]:+.2f} | Trades={r[2]} | WR={r[3]:.0f}%")
else:
    print("\nNo daily_performance rows.")

conn.close()
