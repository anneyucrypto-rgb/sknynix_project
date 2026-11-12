"""Transform raw Parquet data into Granger/VARX and arbitrage feature panels."""

from __future__ import annotations

import argparse
from datetime import date, datetime, time
from pathlib import Path
from typing import Any, Final
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


PROJECT_DIR: Final = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR: Final = PROJECT_DIR / "data" / "raw"
DEFAULT_PROCESSED_DIR: Final = PROJECT_DIR / "data" / "processed"
DEFAULT_ADS_SHARES_PER_ADS: Final = 0.1
NY_TZ: Final = ZoneInfo("America/New_York")
SEOUL_TZ: Final = ZoneInfo("Asia/Seoul")
UTC_TZ: Final = ZoneInfo("UTC")
HORIZONS: Final = (5, 10, 30, 60, 90, 120)
MICROSTRUCTURE_WINDOWS: Final = (1, 5, 10, 30, 60)
BAR_MINUTES: Final = 5
PRE_EVENT_WINDOW_MINUTES: Final = 30
PRE_EVENT_BAR_COUNT: Final = PRE_EVENT_WINDOW_MINUTES // BAR_MINUTES
EVENT_ANCHORS: Final = (
    ("premarket_0400", time(4, 0)),
    ("regular_0930", time(9, 30)),
)

RAW_FILES: Final = {
    "skhy": Path("us/SKHY_5min.parquet"),
    "spy": Path("us/SPY_5min.parquet"),
    "soxx": Path("us/SOXX_5min.parquet"),
    "qqq": Path("us/QQQ_5min.parquet"),
    "usdkrw": Path("fx/USDKRW_5min.parquet"),
    "korea": Path("korea/000660_daily.parquet"),
}
MACRO_FILES: Final = {
    "kospi": Path("macro/KOSPI_daily.parquet"),
    "vix": Path("macro/VIX_daily.parquet"),
    "us_treasury": Path("macro/US_TREASURY_daily.parquet"),
}

TICK_FILES: Final = {
    "trades": Path("us/SKHY_trades.parquet"),
    "quotes": Path("us/SKHY_quotes.parquet"),
}


def _safe_log_ratio(numerator: Any, denominator: Any) -> float:
    if pd.isna(numerator) or pd.isna(denominator):
        return np.nan
    numerator = float(numerator)
    denominator = float(denominator)
    if numerator <= 0 or denominator <= 0:
        return np.nan
    return float(np.log(numerator / denominator))


def _required_paths(raw_dir: Path) -> dict[str, Path]:
    paths = {name: raw_dir / relative for name, relative in RAW_FILES.items()}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Required raw files are missing:\n- " + "\n- ".join(missing)
        )
    return paths


def _load_bars(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path).copy()
    timestamp_column = "timestamp_utc" if "timestamp_utc" in frame else "timestamp"
    frame["timestamp_utc"] = pd.to_datetime(frame[timestamp_column], utc=True)
    if timestamp_column != "timestamp_utc":
        frame = frame.drop(columns=[timestamp_column])
    frame["timestamp_ny"] = frame["timestamp_utc"].dt.tz_convert(NY_TZ)
    frame["event_date_ny"] = frame["timestamp_ny"].dt.date
    frame["event_time_ny"] = frame["timestamp_ny"].dt.time
    return frame.sort_values("timestamp_utc").drop_duplicates("timestamp_utc", keep="last")


def _load_korea(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path).copy()
    frame["trading_date"] = pd.to_datetime(frame["date"]).dt.date
    local_close = pd.to_datetime(frame["trading_date"].astype(str) + " 15:30:00")
    frame["availability_timestamp_utc"] = (
        local_close.dt.tz_localize(SEOUL_TZ).dt.tz_convert(UTC_TZ)
    )
    frame = frame.sort_values("availability_timestamp_utc").reset_index(drop=True)
    frame["prior_adjusted_close"] = frame["adjusted_close"].shift(1)
    frame["kr_return_0"] = np.log(frame["adjusted_close"] / frame["prior_adjusted_close"])
    frame["kr_return_lag1"] = frame["kr_return_0"].shift(1)
    frame["kr_return_lag2"] = frame["kr_return_0"].shift(2)
    frame["korea_range"] = np.log(frame["high"] / frame["low"])
    frame["korea_log_volume"] = np.log1p(frame["volume"])
    return frame


def _load_macro_controls(raw_dir: Path) -> dict[str, pd.DataFrame]:
    paths = {name: raw_dir / relative for name, relative in MACRO_FILES.items()}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Required macro-control files are missing:\n- " + "\n- ".join(missing)
        )

    kospi = pd.read_parquet(paths["kospi"]).copy()
    kospi["trading_date"] = pd.to_datetime(kospi["date"]).dt.date
    kospi_close = pd.to_datetime(kospi["trading_date"].astype(str) + " 15:30:00")
    kospi["availability_timestamp_utc"] = (
        kospi_close.dt.tz_localize(SEOUL_TZ).dt.tz_convert(UTC_TZ)
    )
    kospi["kospi_return_latest"] = np.log(
        kospi["adjusted_close"] / kospi["adjusted_close"].shift(1)
    )

    vix = pd.read_parquet(paths["vix"]).copy()
    vix["trading_date"] = pd.to_datetime(vix["date"]).dt.date
    vix_close = pd.to_datetime(vix["trading_date"].astype(str) + " 16:00:00")
    vix["availability_timestamp_utc"] = (
        vix_close.dt.tz_localize(NY_TZ).dt.tz_convert(UTC_TZ)
    )
    vix["vix_level_lag1d"] = vix["adjusted_close"]
    vix["vix_change_lag1d"] = vix["adjusted_close"].diff()

    treasury = pd.read_parquet(paths["us_treasury"]).copy()
    treasury["trading_date"] = pd.to_datetime(treasury["date"]).dt.date
    treasury_close = pd.to_datetime(
        treasury["trading_date"].astype(str) + " 17:00:00"
    )
    treasury["availability_timestamp_utc"] = (
        treasury_close.dt.tz_localize(NY_TZ).dt.tz_convert(UTC_TZ)
    )
    treasury["us_2y_yield_change_lag1d"] = treasury["us_2y_yield"].diff()
    treasury["us_10y_yield_change_lag1d"] = treasury["us_10y_yield"].diff()

    return {
        "kospi": kospi.sort_values("availability_timestamp_utc").reset_index(drop=True),
        "vix": vix.sort_values("availability_timestamp_utc").reset_index(drop=True),
        "us_treasury": treasury.sort_values("availability_timestamp_utc").reset_index(drop=True),
    }


def _load_optional_events(path: Path, timestamp_column: str) -> pd.DataFrame:
    if not path.is_file() or path.stat().st_size == 0:
        return pd.DataFrame()
    frame = pd.read_parquet(path).copy()
    if frame.empty:
        return frame
    frame["event_date_ny"] = pd.to_datetime(frame["event_date_ny"]).dt.date
    frame["scheduled_timestamp_utc"] = pd.to_datetime(
        frame["scheduled_timestamp_utc"], utc=True
    )
    frame[timestamp_column] = pd.to_datetime(frame[timestamp_column], utc=True)
    return frame


def _load_optional_ticks(path: Path, kind: str) -> tuple[pd.DataFrame, bool]:
    """Load optional full-tick data; false means the source is unavailable.

    These are intentionally different from SKHY_first_trades/quotes.parquet,
    which contain only one observation per event and cannot support window
    counts or averages.
    """
    if not path.is_file() or path.stat().st_size == 0:
        return pd.DataFrame(), False
    frame = pd.read_parquet(path).copy()
    if frame.empty:
        return frame, False

    datetime_candidates = ("timestamp_utc", "sip_datetime", "timestamp")
    timestamp_column = next(
        (column for column in datetime_candidates if column in frame), None
    )
    if timestamp_column is not None:
        frame["timestamp_utc"] = pd.to_datetime(
            frame[timestamp_column], utc=True, errors="coerce"
        )
    elif "sip_timestamp" in frame:
        frame["timestamp_utc"] = pd.to_datetime(
            frame["sip_timestamp"], unit="ns", utc=True, errors="coerce"
        )
    else:
        raise ValueError(f"{path} has no supported tick timestamp column")

    if kind == "trades":
        frame = frame.rename(
            columns={"price": "trade_price", "size": "trade_size"}
        )
        required = {"trade_price", "trade_size"}
    elif kind == "quotes":
        required = {"bid_price", "ask_price", "bid_size", "ask_size"}
    else:
        raise ValueError(f"Unsupported tick kind: {kind}")

    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    for column in required:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame = frame.dropna(subset=["timestamp_utc"]).sort_values("timestamp_utc")
    frame["event_date_ny"] = frame["timestamp_utc"].dt.tz_convert(NY_TZ).dt.date
    if kind == "trades":
        frame = frame[
            (frame["trade_price"] > 0) & (frame["trade_size"] > 0)
        ]
    else:
        frame = frame[
            (frame["bid_price"] > 0)
            & (frame["ask_price"] > 0)
            & (frame["ask_price"] >= frame["bid_price"])
            & (frame["bid_size"] >= 0)
            & (frame["ask_size"] >= 0)
        ].copy()
        frame["midpoint"] = (frame["bid_price"] + frame["ask_price"]) / 2.0
        frame["relative_spread"] = (
            frame["ask_price"] - frame["bid_price"]
        ) / frame["midpoint"]
        depth = frame["bid_size"] + frame["ask_size"]
        frame["quote_imbalance"] = np.where(
            depth > 0,
            (frame["bid_size"] - frame["ask_size"]) / depth,
            np.nan,
        )
    return frame.reset_index(drop=True), True


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    valid = values.notna() & weights.notna() & (weights > 0)
    if not valid.any():
        return np.nan
    return float(np.average(values[valid], weights=weights[valid]))


def _classify_window_trades(
    trades: pd.DataFrame, quotes: pd.DataFrame
) -> pd.DataFrame:
    """Apply the quote rule with a tick-rule fallback and attach midpoints."""
    classified = trades.sort_values("timestamp_utc").copy()
    price_change = classified["trade_price"].diff()
    tick_sign = np.sign(price_change).replace(0, np.nan).ffill()
    classified["trade_sign"] = tick_sign
    classified["midpoint"] = np.nan

    if quotes.empty:
        return classified

    quote_state = quotes[["timestamp_utc", "midpoint"]].sort_values(
        "timestamp_utc"
    )
    classified = pd.merge_asof(
        classified.sort_values("timestamp_utc"),
        quote_state,
        on="timestamp_utc",
        direction="backward",
        tolerance=pd.Timedelta(minutes=5),
        suffixes=("", "_quote"),
    )
    classified["midpoint"] = classified["midpoint_quote"]
    classified = classified.drop(columns=["midpoint_quote"])
    above = classified["trade_price"] > classified["midpoint"]
    below = classified["trade_price"] < classified["midpoint"]
    classified.loc[above, "trade_sign"] = 1.0
    classified.loc[below, "trade_sign"] = -1.0
    return classified


def _tick_window_features(
    trades: pd.DataFrame,
    quotes: pd.DataFrame,
    anchor_timestamp: pd.Timestamp,
    *,
    trades_available: bool,
    quotes_available: bool,
) -> dict[str, Any]:
    """Aggregate post-anchor trades and quotes over nested event windows.

    Windows are half-open: [anchor, anchor + w minutes). Spread and imbalance
    are arithmetic means across valid quote updates. Trade VWAP, effective
    spread, and price impact are share-volume weighted.
    """
    features: dict[str, Any] = {
        "trade_ticks_available": trades_available,
        "quote_ticks_available": quotes_available,
    }
    for window_minutes in MICROSTRUCTURE_WINDOWS:
        suffix = f"0_{window_minutes}m"
        end = anchor_timestamp + pd.Timedelta(minutes=window_minutes)
        trade_window = (
            trades[
                (trades["timestamp_utc"] >= anchor_timestamp)
                & (trades["timestamp_utc"] < end)
            ]
            if trades_available
            else pd.DataFrame()
        )
        quote_window = (
            quotes[
                (quotes["timestamp_utc"] >= anchor_timestamp)
                & (quotes["timestamp_utc"] < end)
            ]
            if quotes_available
            else pd.DataFrame()
        )

        features[f"trade_count_{suffix}"] = (
            int(len(trade_window)) if trades_available else np.nan
        )
        features[f"trade_volume_{suffix}"] = (
            float(trade_window["trade_size"].sum()) if trades_available else np.nan
        )
        features[f"trade_vwap_{suffix}"] = (
            _weighted_mean(trade_window["trade_price"], trade_window["trade_size"])
            if trades_available and not trade_window.empty
            else np.nan
        )
        features[f"quote_count_{suffix}"] = (
            int(len(quote_window)) if quotes_available else np.nan
        )
        features[f"relative_spread_mean_{suffix}"] = (
            float(quote_window["relative_spread"].mean())
            if quotes_available and not quote_window.empty
            else np.nan
        )
        features[f"quote_imbalance_mean_{suffix}"] = (
            float(quote_window["quote_imbalance"].mean())
            if quotes_available and not quote_window.empty
            else np.nan
        )

        if not trades_available or trade_window.empty:
            features[f"signed_volume_{suffix}"] = (
                0.0 if trades_available else np.nan
            )
            features[f"effective_spread_bps_{suffix}"] = np.nan
            features[f"price_impact_bps_{suffix}"] = np.nan
            continue

        quote_history = (
            quotes[
                (quotes["timestamp_utc"] <= end)
                & (
                    quotes["timestamp_utc"]
                    >= anchor_timestamp - pd.Timedelta(minutes=5)
                )
            ]
            if quotes_available
            else pd.DataFrame()
        )
        classified = _classify_window_trades(trade_window, quote_history)
        signed_size = classified["trade_sign"] * classified["trade_size"]
        features[f"signed_volume_{suffix}"] = (
            float(signed_size.sum(min_count=1))
            if signed_size.notna().any()
            else np.nan
        )

        effective_spread = (
            2.0
            * (classified["trade_price"] - classified["midpoint"]).abs()
            / classified["midpoint"]
            * 10_000.0
        )
        features[f"effective_spread_bps_{suffix}"] = _weighted_mean(
            effective_spread, classified["trade_size"]
        )

        if quote_history.empty:
            features[f"price_impact_bps_{suffix}"] = np.nan
        else:
            end_midpoint = float(quote_history.iloc[-1]["midpoint"])
            price_impact = (
                2.0
                * classified["trade_sign"]
                * (end_midpoint - classified["midpoint"])
                / classified["midpoint"]
                * 10_000.0
            )
            features[f"price_impact_bps_{suffix}"] = _weighted_mean(
                price_impact, classified["trade_size"]
            )
    return features


def _date_groups(frame: pd.DataFrame) -> dict[date, pd.DataFrame]:
    return {
        event_date: group.sort_values("timestamp_utc").reset_index(drop=True)
        for event_date, group in frame.groupby("event_date_ny", sort=True)
    }


def _first_bar(group: pd.DataFrame | None, timestamp_utc: pd.Timestamp) -> pd.Series | None:
    if group is None or group.empty:
        return None
    candidates = group[group["timestamp_utc"] >= timestamp_utc]
    if candidates.empty:
        return None
    return candidates.iloc[0]


def _exact_bar(group: pd.DataFrame | None, timestamp_utc: pd.Timestamp) -> pd.Series | None:
    if group is None or group.empty:
        return None
    matches = group[group["timestamp_utc"] == timestamp_utc]
    return None if matches.empty else matches.iloc[-1]


def _latest_at_or_before(
    frame: pd.DataFrame, timestamp_utc: pd.Timestamp
) -> pd.Series | None:
    candidates = frame[frame["timestamp_utc"] <= timestamp_utc]
    return None if candidates.empty else candidates.iloc[-1]


def _latest_available_daily(
    frame: pd.DataFrame, timestamp_utc: pd.Timestamp
) -> pd.Series | None:
    candidates = frame[frame["availability_timestamp_utc"] <= timestamp_utc]
    return None if candidates.empty else candidates.iloc[-1]


def _exact_pre_event_return(
    group: pd.DataFrame | None,
    anchor_bar: pd.Series | None,
    anchor_timestamp: pd.Timestamp,
) -> float:
    """Return the exact five-minute price change ending at the anchor."""
    if group is None or anchor_bar is None:
        return np.nan
    prior_bar = _exact_bar(
        group, anchor_timestamp - pd.Timedelta(minutes=BAR_MINUTES)
    )
    if prior_bar is None:
        return np.nan
    return _safe_log_ratio(anchor_bar.get("open"), prior_bar.get("close"))


def _previous_regular_close(
    groups: dict[date, pd.DataFrame], event_date: date
) -> pd.Series | None:
    previous_dates = [candidate for candidate in groups if candidate < event_date]
    if not previous_dates:
        return None
    group = groups[max(previous_dates)]
    regular = group[
        (group["event_time_ny"] >= time(9, 30))
        & (group["event_time_ny"] <= time(15, 59, 59))
    ]
    return None if regular.empty else regular.iloc[-1]


def _pre_event_features(
    group: pd.DataFrame | None,
    anchor_timestamp: pd.Timestamp,
    anchor_bar: pd.Series | None,
    previous_close: pd.Series | None,
    anchor_type: str,
) -> dict[str, Any]:
    if anchor_bar is None:
        return {
            "return": np.nan,
            "rv_30m": np.nan,
            "log_volume_30m": np.nan,
            "vwap_deviation": np.nan,
            "window_available": False,
        }

    if group is None:
        group = pd.DataFrame()
    window = group[
        (group["timestamp_utc"] >= anchor_timestamp - pd.Timedelta(minutes=30))
        & (group["timestamp_utc"] < anchor_timestamp)
    ].sort_values("timestamp_utc")

    if anchor_type == "regular_0930" and not window.empty:
        pre_return = _safe_log_ratio(anchor_bar.get("open"), window.iloc[0].get("open"))
    elif anchor_type == "premarket_0400" and previous_close is not None:
        pre_return = _safe_log_ratio(anchor_bar.get("open"), previous_close.get("close"))
    else:
        pre_return = np.nan

    returns = np.log(window["close"]).diff() if not window.empty else pd.Series(dtype=float)
    realized_volatility = (
        float(np.sqrt(np.nansum(np.square(returns))))
        if len(window) == PRE_EVENT_BAR_COUNT
        else np.nan
    )
    log_volume = (
        float(np.log1p(window["volume"].sum()))
        if len(window) == PRE_EVENT_BAR_COUNT and "volume" in window
        else np.nan
    )
    reference = window.iloc[-1] if not window.empty else previous_close
    vwap_deviation = (
        _safe_log_ratio(reference.get("close"), reference.get("vwap"))
        if reference is not None
        else np.nan
    )
    return {
        "return": pre_return,
        "rv_30m": realized_volatility,
        "log_volume_30m": log_volume,
        "vwap_deviation": vwap_deviation,
        "window_available": len(window) == PRE_EVENT_BAR_COUNT,
    }


def _future_return(
    group: pd.DataFrame | None,
    anchor_bar: pd.Series | None,
    horizon_minutes: int,
) -> tuple[float, pd.Timestamp | pd.NaT, float]:
    """Return log(P[t0+h] / P[t0]), target timestamp, and P[t0+h].

    P[t0] is the opening price of the first qualifying five-minute bar at or
    after the scheduled event anchor. P[t0+h] is the close of the five-minute
    bar ending exactly h minutes after that actual anchor.
    """
    if anchor_bar is None:
        return np.nan, pd.NaT, np.nan
    target_start = anchor_bar["timestamp_utc"] + pd.Timedelta(
        minutes=horizon_minutes - BAR_MINUTES
    )
    target = _exact_bar(group, target_start)
    if target is None:
        return np.nan, pd.NaT, np.nan
    target_price = target.get("close")
    return (
        _safe_log_ratio(target_price, anchor_bar.get("open")),
        target["timestamp_utc"] + pd.Timedelta(minutes=BAR_MINUTES),
        float(target_price) if pd.notna(target_price) else np.nan,
    )


def _event_lookup(frame: pd.DataFrame) -> dict[tuple[date, str], pd.Series]:
    if frame.empty:
        return {}
    return {
        (row["event_date_ny"], row["anchor_type"]): row
        for _, row in frame.iterrows()
    }


def build_granger_panel(
    bars: dict[str, pd.DataFrame],
    korea: pd.DataFrame,
    macro: dict[str, pd.DataFrame],
    trades: pd.DataFrame,
    quotes: pd.DataFrame,
    tick_trades: pd.DataFrame,
    tick_quotes: pd.DataFrame,
    *,
    tick_trades_available: bool,
    tick_quotes_available: bool,
) -> pd.DataFrame:
    """Create one wide feature row per U.S. event date and anchor."""
    groups = {name: _date_groups(frame) for name, frame in bars.items() if name != "usdkrw"}
    skhy_groups = groups["skhy"]
    trade_lookup = _event_lookup(trades)
    quote_lookup = _event_lookup(quotes)
    tick_trade_groups = (
        _date_groups(tick_trades) if tick_trades_available else {}
    )
    tick_quote_groups = (
        _date_groups(tick_quotes) if tick_quotes_available else {}
    )
    rows: list[dict[str, Any]] = []

    for event_date in sorted(skhy_groups):
        for anchor_type, anchor_time in EVENT_ANCHORS:
            scheduled_local = datetime.combine(event_date, anchor_time, NY_TZ)
            scheduled_utc = pd.Timestamp(scheduled_local.astimezone(UTC_TZ))
            key = (event_date, anchor_type)

            anchor_bars = {
                name: _first_bar(groups[name].get(event_date), scheduled_utc)
                for name in ("skhy", "spy", "soxx", "qqq")
            }
            skhy_anchor = anchor_bars["skhy"]
            if skhy_anchor is None:
                continue

            previous_closes = {
                name: _previous_regular_close(groups[name], event_date)
                for name in ("skhy", "spy", "soxx", "qqq")
            }
            korea_candidates = korea[
                korea["availability_timestamp_utc"] <= scheduled_utc
            ]
            korea_row = None if korea_candidates.empty else korea_candidates.iloc[-1]
            trade_row = trade_lookup.get(key)
            quote_row = quote_lookup.get(key)

            row: dict[str, Any] = {
                "event_date_ny": event_date,
                "anchor_type": anchor_type,
                "timestamp_utc": scheduled_utc,
                "t_first_minute_bar": skhy_anchor["timestamp_utc"],
                "t_first_trade": trade_row.get("t_first_trade") if trade_row is not None else pd.NaT,
                "t_first_quote": quote_row.get("t_first_quote") if quote_row is not None else pd.NaT,
                "first_trade_price": trade_row.get("trade_price") if trade_row is not None else np.nan,
                "first_trade_size": trade_row.get("trade_size") if trade_row is not None else np.nan,
                "first_bid_price": quote_row.get("bid_price") if quote_row is not None else np.nan,
                "first_ask_price": quote_row.get("ask_price") if quote_row is not None else np.nan,
                "first_bid_size": quote_row.get("bid_size") if quote_row is not None else np.nan,
                "first_ask_size": quote_row.get("ask_size") if quote_row is not None else np.nan,
                "skhy_anchor_price": skhy_anchor.get("open"),
                "skhy_anchor_open": skhy_anchor.get("open"),
                "skhy_anchor_close": skhy_anchor.get("close"),
                "skhy_anchor_volume": skhy_anchor.get("volume"),
                "previous_us_regular_close": (
                    previous_closes["skhy"].get("close")
                    if previous_closes["skhy"] is not None
                    else np.nan
                ),
                "exact_minute_anchor": skhy_anchor["timestamp_utc"] == scheduled_utc,
                "iso_day_of_week": event_date.isoweekday(),
            }

            if korea_row is not None:
                row.update(
                    {
                        "korea_trading_date": korea_row["trading_date"],
                        "korea_availability_timestamp_utc": korea_row[
                            "availability_timestamp_utc"
                        ],
                        "korea_open": korea_row["open"],
                        "korea_high": korea_row["high"],
                        "korea_low": korea_row["low"],
                        "korea_close": korea_row["close"],
                        "korea_adjusted_close": korea_row["adjusted_close"],
                        "korea_prior_adjusted_close": korea_row[
                            "prior_adjusted_close"
                        ],
                        "korea_volume": korea_row["volume"],
                        "kr_return_0": korea_row["kr_return_0"],
                        "kr_return_lag1": korea_row["kr_return_lag1"],
                        "kr_return_lag2": korea_row["kr_return_lag2"],
                        "korea_range": korea_row["korea_range"],
                        "korea_log_volume": korea_row["korea_log_volume"],
                        "korea_stale_days": (event_date - korea_row["trading_date"]).days,
                        "korea_new_close": event_date == korea_row["trading_date"],
                    }
                )

            skhy_pre = _pre_event_features(
                skhy_groups.get(event_date),
                scheduled_utc,
                skhy_anchor,
                previous_closes["skhy"],
                anchor_type,
            )
            row.update(
                {
                    "skhy_pre_event_return": skhy_pre["return"],
                    "skhy_pre_event_rv_30m": skhy_pre["rv_30m"],
                    "skhy_pre_event_log_volume_30m": skhy_pre["log_volume_30m"],
                    "skhy_vwap_deviation": skhy_pre["vwap_deviation"],
                    "pre_event_window_available": skhy_pre["window_available"],
                    "overnight_gap": _safe_log_ratio(
                        skhy_anchor.get("open"),
                        previous_closes["skhy"].get("close")
                        if previous_closes["skhy"] is not None
                        else np.nan,
                    ),
                }
            )

            for name in ("spy", "soxx", "qqq"):
                feature = _pre_event_features(
                    groups[name].get(event_date),
                    scheduled_utc,
                    anchor_bars[name],
                    previous_closes[name],
                    anchor_type,
                )
                row[f"{name}_pre_event_return"] = feature["return"]
                row[f"{name}_pre_event_return_5m"] = _exact_pre_event_return(
                    groups[name].get(event_date),
                    anchor_bars[name],
                    scheduled_utc,
                )

            fx_now = _latest_at_or_before(bars["usdkrw"], scheduled_utc)
            fx_pre = _latest_at_or_before(
                bars["usdkrw"], scheduled_utc - pd.Timedelta(minutes=30)
            )
            row["usdkrw_return_latest"] = _safe_log_ratio(
                fx_now.get("close") if fx_now is not None else np.nan,
                fx_pre.get("close") if fx_pre is not None else np.nan,
            )

            kospi_row = _latest_available_daily(macro["kospi"], scheduled_utc)
            vix_row = _latest_available_daily(macro["vix"], scheduled_utc)
            treasury_row = _latest_available_daily(
                macro["us_treasury"], scheduled_utc
            )
            row["kospi_return_latest"] = (
                kospi_row.get("kospi_return_latest")
                if kospi_row is not None
                else np.nan
            )
            row["vix_level_lag1d"] = (
                vix_row.get("vix_level_lag1d") if vix_row is not None else np.nan
            )
            row["vix_change_lag1d"] = (
                vix_row.get("vix_change_lag1d") if vix_row is not None else np.nan
            )
            row["us_2y_yield_change_lag1d"] = (
                treasury_row.get("us_2y_yield_change_lag1d")
                if treasury_row is not None
                else np.nan
            )
            row["us_10y_yield_change_lag1d"] = (
                treasury_row.get("us_10y_yield_change_lag1d")
                if treasury_row is not None
                else np.nan
            )

            row.update(
                _tick_window_features(
                    tick_trade_groups.get(event_date, pd.DataFrame()),
                    tick_quote_groups.get(event_date, pd.DataFrame()),
                    skhy_anchor["timestamp_utc"],
                    trades_available=tick_trades_available,
                    quotes_available=tick_quotes_available,
                )
            )

            for horizon in HORIZONS:
                for name in ("skhy", "spy", "soxx", "qqq"):
                    value, target_time, target_price = _future_return(
                        groups[name].get(event_date), anchor_bars[name], horizon
                    )
                    row[f"{name}_fwd_return_{horizon}m"] = value
                    if name == "skhy":
                        row[f"skhy_target_price_{horizon}m"] = target_price
                        row[f"skhy_target_timestamp_{horizon}m"] = target_time
                        row[f"skhy_fwd_return_{horizon}m_missing"] = pd.isna(value)
                        total_response = _safe_log_ratio(
                            target_price, row["previous_us_regular_close"]
                        )
                        row[f"skhy_total_response_{horizon}m"] = total_response
                        row[f"skhy_total_response_{horizon}m_missing"] = pd.isna(
                            total_response
                        )
                fx_target = _latest_at_or_before(
                    bars["usdkrw"],
                    scheduled_utc + pd.Timedelta(minutes=horizon),
                )
                row[f"usdkrw_fwd_return_{horizon}m"] = _safe_log_ratio(
                    fx_target.get("close") if fx_target is not None else np.nan,
                    fx_now.get("close") if fx_now is not None else np.nan,
                )

            forward_missing = [
                row[f"skhy_fwd_return_{horizon}m_missing"]
                for horizon in HORIZONS
            ]
            total_missing = [
                row[f"skhy_total_response_{horizon}m_missing"]
                for horizon in HORIZONS
            ]
            row["missing_target_count"] = int(sum(forward_missing))
            row["all_targets_complete"] = not any(forward_missing)
            row["missing_total_response_count"] = int(sum(total_missing))
            row["all_total_responses_complete"] = not any(total_missing)

            rows.append(row)

    panel = pd.DataFrame.from_records(rows).sort_values(
        ["anchor_type", "event_date_ny"]
    )
    panel["skhy_pre_event_volume_z"] = panel.groupby("anchor_type")[
        "skhy_pre_event_log_volume_30m"
    ].transform(lambda series: (series - series.mean()) / series.std(ddof=1))

    lag_targets = [
        *(f"skhy_fwd_return_{horizon}m" for horizon in HORIZONS),
        *(f"skhy_total_response_{horizon}m" for horizon in HORIZONS),
    ]
    for column in lag_targets:
        panel[f"{column}_lag1"] = panel.groupby("anchor_type")[column].shift(1)
        panel[f"{column}_lag2"] = panel.groupby("anchor_type")[column].shift(2)

    kr_expected = np.log(
        panel["korea_adjusted_close"] / panel["korea_prior_adjusted_close"]
    )
    kr_complete = panel["kr_return_0"].notna() & kr_expected.notna()
    if not np.allclose(
        panel.loc[kr_complete, "kr_return_0"],
        kr_expected.loc[kr_complete],
        rtol=1e-12,
        atol=1e-12,
    ):
        raise AssertionError("Korean return identity failed")

    for horizon in HORIZONS:
        total = panel[f"skhy_total_response_{horizon}m"]
        decomposed = panel["overnight_gap"] + panel[
            f"skhy_fwd_return_{horizon}m"
        ]
        complete = total.notna() & decomposed.notna()
        if not np.allclose(
            total.loc[complete],
            decomposed.loc[complete],
            rtol=1e-12,
            atol=1e-12,
        ):
            raise AssertionError(
                f"Total-response decomposition failed at {horizon} minutes"
            )

    return panel.sort_values(["event_date_ny", "anchor_type"]).reset_index(drop=True)


def build_arbitrage_panel(
    bars: dict[str, pd.DataFrame],
    korea: pd.DataFrame,
    granger_panel: pd.DataFrame,
    ads_shares_per_ads: float,
) -> pd.DataFrame:
    """Create a five-minute panel with FX parity, premium, and residual features."""
    skhy = bars["skhy"].copy().sort_values("timestamp_utc")
    fx = bars["usdkrw"][["timestamp_utc", "close"]].rename(
        columns={"close": "usdkrw_close"}
    )
    korea_asof = korea[
        [
            "availability_timestamp_utc",
            "trading_date",
            "close",
            "adjusted_close",
            "kr_return_0",
            "kr_return_lag1",
            "korea_range",
            "volume",
        ]
    ].rename(
        columns={
            "close": "korea_close",
            "adjusted_close": "korea_adjusted_close",
            "volume": "korea_volume",
        }
    )

    panel = pd.merge_asof(
        skhy,
        korea_asof.sort_values("availability_timestamp_utc"),
        left_on="timestamp_utc",
        right_on="availability_timestamp_utc",
        direction="backward",
    )
    panel = pd.merge_asof(
        panel.sort_values("timestamp_utc"),
        fx.sort_values("timestamp_utc"),
        on="timestamp_utc",
        direction="backward",
        tolerance=pd.Timedelta(minutes=5),
    )
    panel["ads_shares_per_ads"] = ads_shares_per_ads
    panel["korea_parity_usd_per_ads"] = (
        panel["korea_adjusted_close"]
        * ads_shares_per_ads
        / panel["usdkrw_close"]
    )
    panel["arbitrage_premium_log"] = np.log(
        panel["close"] / panel["korea_parity_usd_per_ads"]
    )
    panel["arbitrage_premium_pct"] = np.expm1(panel["arbitrage_premium_log"])
    panel["skhy_return_5m"] = np.log(panel["close"]).diff()
    panel["usdkrw_return_5m"] = np.log(panel["usdkrw_close"]).diff()

    for window in (30, 60, 120, 390):
        observations = window // BAR_MINUTES
        rolling = panel["arbitrage_premium_log"].rolling(
            observations, min_periods=max(2, observations // 4)
        )
        mean = rolling.mean()
        std = rolling.std(ddof=1)
        panel[f"premium_residual_{window}m"] = panel["arbitrage_premium_log"] - mean
        panel[f"premium_zscore_{window}m"] = (
            panel["arbitrage_premium_log"] - mean
        ) / std.replace(0, np.nan)

    local_minutes = panel["timestamp_ny"].dt.hour * 60 + panel["timestamp_ny"].dt.minute
    panel["minutes_from_0400"] = local_minutes - 4 * 60
    panel["minutes_from_0930"] = local_minutes - (9 * 60 + 30)
    panel["session"] = np.select(
        [
            (local_minutes >= 4 * 60) & (local_minutes < 9 * 60 + 30),
            (local_minutes >= 9 * 60 + 30) & (local_minutes < 16 * 60),
            (local_minutes >= 16 * 60) & (local_minutes <= 20 * 60),
        ],
        ["premarket", "regular", "afterhours"],
        default="outside",
    )
    panel["is_0400_event"] = local_minutes == 4 * 60
    panel["is_0930_event"] = local_minutes == 9 * 60 + 30

    timestamp_columns = ["t_first_trade", "t_first_quote", "t_first_minute_bar"]
    for anchor_type, prefix in (
        ("premarket_0400", "premarket"),
        ("regular_0930", "regular"),
    ):
        event_timestamps = granger_panel[
            granger_panel["anchor_type"] == anchor_type
        ][["event_date_ny", *timestamp_columns]].rename(
            columns={column: f"{prefix}_{column}" for column in timestamp_columns}
        )
        panel = panel.merge(event_timestamps, on="event_date_ny", how="left")

    return panel.sort_values("timestamp_utc").reset_index(drop=True)


def build_processed_panels(
    raw_dir: Path = DEFAULT_RAW_DIR,
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
    ads_shares_per_ads: float = DEFAULT_ADS_SHARES_PER_ADS,
) -> dict[str, int]:
    """Read raw files and atomically write the two processed feature panels."""
    if ads_shares_per_ads <= 0:
        raise ValueError("ads_shares_per_ads must be positive")
    raw_dir = raw_dir.resolve()
    processed_dir = processed_dir.resolve()
    processed_dir.mkdir(parents=True, exist_ok=True)
    paths = _required_paths(raw_dir)
    bars = {name: _load_bars(path) for name, path in paths.items() if name != "korea"}
    korea = _load_korea(paths["korea"])
    macro = _load_macro_controls(raw_dir)
    trades = _load_optional_events(
        raw_dir / "us" / "SKHY_first_trades.parquet", "t_first_trade"
    )
    quotes = _load_optional_events(
        raw_dir / "us" / "SKHY_first_quotes.parquet", "t_first_quote"
    )
    tick_trades, tick_trades_available = _load_optional_ticks(
        raw_dir / TICK_FILES["trades"], "trades"
    )
    tick_quotes, tick_quotes_available = _load_optional_ticks(
        raw_dir / TICK_FILES["quotes"], "quotes"
    )

    granger_panel = build_granger_panel(
        bars,
        korea,
        macro,
        trades,
        quotes,
        tick_trades,
        tick_quotes,
        tick_trades_available=tick_trades_available,
        tick_quotes_available=tick_quotes_available,
    )
    arbitrage_panel = build_arbitrage_panel(
        bars, korea, granger_panel, ads_shares_per_ads
    )
    granger_path = processed_dir / "granger_panel.parquet"
    arbitrage_path = processed_dir / "arbitrage_panel_5min.parquet"
    _atomic_parquet(granger_panel, granger_path)
    _atomic_parquet(arbitrage_panel, arbitrage_path)
    return {
        str(granger_path): len(granger_panel),
        str(arbitrage_path): len(arbitrage_panel),
    }


def _atomic_parquet(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    frame.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(destination)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument(
        "--ads-shares-per-ads",
        type=float,
        default=DEFAULT_ADS_SHARES_PER_ADS,
        help="Korean common shares represented by one SKHY ADS (default: 0.1).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    counts = build_processed_panels(
        args.raw_dir, args.processed_dir, args.ads_shares_per_ads
    )
    for path, count in counts.items():
        print(f"{path}: {count:,} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
