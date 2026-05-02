from core.data_pipeline import DataPipeline

d = DataPipeline()

df = d.get_daily_bars("AAPL")
df_ind = d.compute_indicators(df)
ind_cols = [c for c in df_ind.columns if c not in ["open", "high", "low", "close", "volume"]]
print("Indicators:", ind_cols)

feat = d.get_features("AAPL")
print("Features sample:", {k: feat[k] for k in ["close", "rsi_14", "macd", "vol_ratio"]})

regime = d.get_market_regime()
print("Market regime (SPY):", regime)

stats = d.get_universe_stats(["AAPL", "MSFT", "NVDA"])
print("Universe stats:")
print(stats)
