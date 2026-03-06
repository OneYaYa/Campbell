from __future__ import annotations

import sys
import pandas as pd

from sentiment_strategy import (
    SentimentDirectionalConfig,
    build_and_backtest_one_instrument,
)

WTI_CSV = "data/wti.csv"
BRENT_CSV = "data/brent.csv"


def main() -> None:
    # Check for sentiment CSV argument
    sentiment_csv = None
    if len(sys.argv) > 1:
        sentiment_csv = sys.argv[1]
        print(f"Using sentiment data from: {sentiment_csv}")
    else:
        print("Usage: python backtest_runner.py [sentiment_csv_path]")
        print("Running with mock sentiment data (no sentiment_csv provided)...")
    
    # You can tune thresholds here, but keep defaults for now
    cfg = SentimentDirectionalConfig(
        entry_z=0.5,
        exit_z=0.1,
        target_vol=0.01,
        w_max=2.0,
        cost_per_unit=0.0002,
        enable_vol_spike_filter=True,
        vol_spike_mult=2.5,
        next_day_execution=True,
    )

    # Different seeds so WTI/Brent get different mock sentiment paths
    # (only used if sentiment_csv is not provided)
    bt_wti, stats_wti = build_and_backtest_one_instrument(
        WTI_CSV, seed=7, cfg=cfg, sentiment_csv=sentiment_csv
    )
    bt_brent, stats_brent = build_and_backtest_one_instrument(
        BRENT_CSV, seed=11, cfg=cfg, sentiment_csv=sentiment_csv
    )

    print("\n=== WTI Strategy Performance ===")
    for k, v in stats_wti.items():
        print(f"{k:>20}: {v}")

    print("\n=== Brent Strategy Performance ===")
    for k, v in stats_brent.items():
        print(f"{k:>20}: {v}")

    # Save results for inspection
    bt_wti.to_csv("data/wti_strategy_backtest.csv", index=False)
    bt_brent.to_csv("data/brent_strategy_backtest.csv", index=False)

    print("\nSaved:")
    print("  data/wti_strategy_backtest.csv")
    print("  data/brent_strategy_backtest.csv")

    # Optional quick sanity check: show last few lines
    print("\nWTI tail:")
    print(bt_wti.tail(5)[["date", "Z", "pos_state", "weight", "ret", "strat_ret", "equity"]])

    print("\nBrent tail:")
    print(bt_brent.tail(5)[["date", "Z", "pos_state", "weight", "ret", "strat_ret", "equity"]])


if __name__ == "__main__":
    main()
