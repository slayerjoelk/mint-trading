import json
import os
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_training_mode():
    """Force each equity agent to emit at least 1 buy signal on a random universe ticker.
    Bypasses strict thresholds. Uses qty=1 market orders for pipeline verification.
    """
    import sys
    sys.path.insert(0, ROOT)
    from core.db import DatabaseManager
    from core.market import MarketInterface
    from core.data_pipeline import DataPipeline
    from core.risk_manager import RiskManager

    db = DatabaseManager()
    db.create_tables()
    market = MarketInterface()
    data = DataPipeline()
    risk = RiskManager(market, db)

    print("\n=== TRAINING MODE — Pipeline Verification ===\n")

    agents = [
        ("Athena", "mean_reversion", ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "JPM", "V", "JNJ", "PG"]),
        ("Orion", "momentum", ["NVDA", "AMD", "TSLA", "META", "NFLX", "CRM", "ADBE", "SHOP", "SNOW", "PLTR"]),
        ("Sibyl", "event", ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "NFLX", "CRM", "ADBE"]),
        ("Janus", "defensive", ["JNJ", "PG", "WMT", "VZ", "KO", "PEP", "XLU", "XLF", "GLD", "TLT"]),
    ]

    results = []
    clock = market.get_clock()
    is_open = clock.get("is_open", True)

    if not is_open:
        print("⚠ Market closed — training mode skips execution.")
        return

    for name, style, universe in agents:
        # Pick first ticker with valid features
        ticker = None
        features = None
        for t in universe:
            f = data.get_features(t)
            if f and f.get("close"):
                ticker = t
                features = f
                break

        if not ticker:
            print(f"  {name}: ✗ No data available for any ticker")
            results.append((name, None, "no_data"))
            continue

        close = features["close"]
        qty = 1  # Training mode: exactly 1 share

        # Simulate a "training signal" with low confidence
        result = risk.validate_trade(f"training-{name}", ticker, qty, "buy", close)
        if result.get("approved"):
            aqty = result.get("adjusted_quantity", qty)
            order = market.submit_order(ticker, aqty, "buy", "market")
            if "error" not in order:
                results.append((name, ticker, "ok"))
                print(f"  {name}: ✓ BUY {aqty} {ticker} @ ${close:.2f} ({style}) — TRAINING")
            else:
                results.append((name, ticker, f"order_error: {order.get('error')}"))
                print(f"  {name}: ✗ BUY {aqty} {ticker} — ORDER ERROR: {order.get('error')}")
        else:
            results.append((name, ticker, f"blocked: {result.get('reason')}"))
            print(f"  {name}: ✗ BUY {qty} {ticker} — BLOCKED: {result.get('reason')}")

    # Log to training log
    log_path = os.path.join(ROOT, ".hermes", "training-log.jsonl")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "mode": "training",
        "trades": [{"agent": r[0], "ticker": r[1], "status": r[2]} for r in results],
        "market_open": is_open,
    }
    with open(log_path, "a") as f:
        f.write(json.dumps(entry) + "\n")

    print(f"\n=== Training Complete: {sum(1 for r in results if r[2] == 'ok')}/{len(agents)} agents traded ===\n")
    return results
