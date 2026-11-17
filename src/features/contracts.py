"""Typed contracts for the six principal research datasets."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetContract:
    name: str
    grain: str
    primary_key: tuple[str, ...]
    required_columns: tuple[str, ...]
    timestamp_columns: tuple[str, ...] = ()
    description: str = ""


CONTRACTS: dict[str, DatasetContract] = {
    "daily": DatasetContract(
        "daily",
        "one paired U.S. trading session",
        ("date_us",),
        (
            "date_us", "us_close_ts_utc", "adr_close_usd", "kr_session_date",
            "kr_close_ts_utc", "kr_available_at_utc", "kr_close_krw",
            "fx_asof_ts_utc", "usdkrw_asof", "log_adr", "log_kr_usd",
            "parity_spread", "adr_daily_return", "kr_daily_return_known",
            "us_sector_return", "kr_sector_return_known", "data_quality_flags",
        ),
        ("us_close_ts_utc", "kr_close_ts_utc", "kr_available_at_utc", "fx_asof_ts_utc"),
    ),
    "us_5min": DatasetContract(
        "us_5min",
        "one completed SKHY five-minute interval",
        ("date_us", "bar_end_utc"),
        (
            "date_us", "bar_start_utc", "bar_end_utc", "session_type",
            "open", "high", "low", "close", "volume", "trade_count",
            "dollar_volume", "bid", "ask", "mid", "bid_size", "ask_size",
            "spread_bps", "quote_updates", "quote_coverage", "quote_age_seconds",
            "fx_mid_asof", "kr_close_known", "parity_spread", "valid_price",
            "valid_quote",
        ),
        ("bar_start_utc", "bar_end_utc"),
    ),
    "anchors": DatasetContract(
        "anchors",
        "one U.S. date by anchor type",
        ("date_us", "anchor_type"),
        (
            "date_us", "anchor_type", "anchor_ts_utc", "anchor_valid",
            "selection_status", "anchor_mid_usd", "anchor_trade_usd",
            "liquidity_score", "spread_bps", "volume_5min", "trade_count_5min",
            "quote_updates_5min", "kr_close_known", "usdkrw_anchor",
            "parity_spread_anchor", "selection_version",
        ),
        ("anchor_ts_utc",),
    ),
    "h1": DatasetContract(
        "h1",
        "one eligible daily observation",
        ("date_us",),
        (
            "date_us", "log_adr", "log_kr_usd", "d_log_adr", "d_log_kr_usd",
            "log_adr_l1", "log_kr_usd_l1", "d_log_adr_l1", "d_log_kr_usd_l1",
            "sample_eligible",
        ),
    ),
    "h2": DatasetContract(
        "h2",
        "one U.S. date by anchor and horizon",
        ("date_us", "anchor_type", "horizon_min"),
        (
            "date_us", "anchor_type", "anchor_ts_utc", "horizon_min",
            "endpoint_ts_utc", "ect_anchor", "liquidity_score",
            "ect_x_liquidity", "spread_bps", "kr_return_known", "adr_return_h",
            "ect_change_h", "crosses_open", "valid_outcome", "h1_model_id",
        ),
        ("anchor_ts_utc", "endpoint_ts_utc"),
    ),
    "h3": DatasetContract(
        "h3",
        "one directional target observation",
        ("direction", "source_session_date", "target_session_date", "anchor_type", "horizon_min"),
        (
            "direction", "source_session_date", "target_session_date",
            "information_cutoff_utc", "anchor_type", "horizon_min",
            "ect_pre_response", "source_market_return", "target_market_return",
            "valid_outcome", "h1_model_id",
        ),
        ("information_cutoff_utc",),
    ),
}
