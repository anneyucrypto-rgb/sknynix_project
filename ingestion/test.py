"""Small smoke test for processed Parquet outputs."""

from pathlib import Path

import pandas as pd

from database import _tick_window_features


PROJECT = Path(__file__).resolve().parents[1]
HORIZONS = (5, 10, 30, 60, 90, 120)
MICROSTRUCTURE_WINDOWS = (1, 5, 10, 30, 60)
MICROSTRUCTURE_METRICS = (
    "trade_count",
    "trade_volume",
    "trade_vwap",
    "signed_volume",
    "quote_count",
    "relative_spread_mean",
    "quote_imbalance_mean",
    "effective_spread_bps",
    "price_impact_bps",
)


def test_tick_window_features() -> None:
    anchor = pd.Timestamp("2026-01-02 14:30:00", tz="UTC")
    trades = pd.DataFrame(
        {
            "timestamp_utc": [
                anchor + pd.Timedelta(seconds=10),
                anchor + pd.Timedelta(seconds=70),
            ],
            "trade_price": [100.10, 99.90],
            "trade_size": [10.0, 20.0],
        }
    )
    quotes = pd.DataFrame(
        {
            "timestamp_utc": [anchor, anchor + pd.Timedelta(minutes=4)],
            "bid_price": [99.90, 100.00],
            "ask_price": [100.10, 100.20],
            "bid_size": [100.0, 120.0],
            "ask_size": [200.0, 80.0],
            "midpoint": [100.00, 100.10],
            "relative_spread": [0.002, 0.2 / 100.1],
            "quote_imbalance": [-1.0 / 3.0, 0.2],
        }
    )
    features = _tick_window_features(
        trades,
        quotes,
        anchor,
        trades_available=True,
        quotes_available=True,
    )
    assert features["trade_count_0_1m"] == 1
    assert features["trade_volume_0_1m"] == 10.0
    assert features["signed_volume_0_1m"] == 10.0
    assert features["quote_count_0_1m"] == 1
    assert features["trade_count_0_5m"] == 2
    assert features["trade_volume_0_5m"] == 30.0
    assert features["signed_volume_0_5m"] == -10.0
    assert abs(features["effective_spread_bps_0_1m"] - 20.0) < 1e-9


def main() -> None:
    test_tick_window_features()
    granger = pd.read_parquet(PROJECT / "data/processed/granger_panel.parquet")
    arbitrage = pd.read_parquet(
        PROJECT / "data/processed/arbitrage_panel_5min.parquet"
    )
    required = {
        *(f"skhy_fwd_return_{horizon}m" for horizon in HORIZONS),
        "t_first_trade",
        "t_first_quote",
        "t_first_minute_bar",
        "trade_ticks_available",
        "quote_ticks_available",
        "spy_pre_event_return_5m",
        "soxx_pre_event_return_5m",
        "qqq_pre_event_return_5m",
        "kospi_return_latest",
        "vix_level_lag1d",
        "vix_change_lag1d",
        "us_2y_yield_change_lag1d",
        "us_10y_yield_change_lag1d",
        *(
            f"{metric}_0_{window}m"
            for metric in MICROSTRUCTURE_METRICS
            for window in MICROSTRUCTURE_WINDOWS
        ),
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
