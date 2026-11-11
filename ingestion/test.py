"""Small smoke test for processed Parquet outputs."""

from pathlib import Path

import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
HORIZONS = (5, 10, 30, 60, 90, 120)


def main() -> None:
    granger = pd.read_parquet(PROJECT / "data/processed/granger_panel.parquet")
    arbitrage = pd.read_parquet(
        PROJECT / "data/processed/arbitrage_panel_1min.parquet"
    )
    required = {
        *(f"skhy_fwd_return_{horizon}m" for horizon in HORIZONS),
        "t_first_trade",
        "t_first_quote",
        "t_first_minute_bar",
    }
    missing = required.difference(granger.columns)
    if missing:
        raise AssertionError(f"granger panel is missing columns: {sorted(missing)}")
    if "arbitrage_premium_log" not in arbitrage:
        raise AssertionError("arbitrage panel is missing arbitrage_premium_log")
    print(f"granger rows: {len(granger):,}")
    print(f"arbitrage rows: {len(arbitrage):,}")


if __name__ == "__main__":
    main()
