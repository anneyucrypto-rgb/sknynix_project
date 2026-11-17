"""Causal dynamic-premarket and fixed-opening anchor selection."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timedelta

import numpy as np
import pandas as pd


NY_TZ = "America/New_York"


@dataclass(frozen=True)
class AnchorSpec:
    start_et: time = time(7, 0)
    cutoff_et: time = time(9, 20)
    fixed_open_et: time = time(9, 30)
    bar_minutes: int = 5
    consecutive_bars: int = 2
    min_trades: int = 10
    min_volume: int = 1_000
    min_quote_updates: int = 10
    max_spread_bps: float = 100.0
    min_quote_coverage: float = 0.80
    max_quote_age_seconds: float = 30.0
    min_liquidity_score: float = 0.0
    version: str = "v1"

    def validate(self) -> None:
        if self.bar_minutes <= 0 or self.consecutive_bars <= 0:
            raise ValueError("bar_minutes and consecutive_bars must be positive")
        if self.start_et >= self.cutoff_et:
            raise ValueError("anchor start must precede cutoff")
        if not 0 <= self.min_quote_coverage <= 1:
            raise ValueError("min_quote_coverage must be between zero and one")


def liquidity_score(frame: pd.DataFrame) -> pd.Series:
    """Deterministic monotone score; historical standardization can replace it."""

    return (
        np.log1p(frame["volume"].clip(lower=0))
        + np.log1p(frame["trade_count"].clip(lower=0))
        + np.log1p(frame["quote_updates"].clip(lower=0))
        + frame["quote_coverage"].fillna(0)
        - np.log1p(frame["spread_bps"].clip(lower=0).fillna(np.inf))
    )


def _anchor_record(
    date_us: object,
    anchor_type: str,
    row: pd.Series | None,
    status: str,
    spec: AnchorSpec,
) -> dict[str, object]:
    valid = row is not None
    return {
        "date_us": date_us,
        "anchor_type": anchor_type,
        "anchor_ts_utc": row["anchor_ts_utc"] if valid else pd.NaT,
        "anchor_valid": valid,
        "selection_status": status,
        "anchor_mid_usd": row.get("mid", np.nan) if valid else np.nan,
        "anchor_trade_usd": row.get("anchor_trade_usd", row.get("close", np.nan)) if valid else np.nan,
        "liquidity_score": row.get("liquidity_score", np.nan) if valid else np.nan,
        "spread_bps": row.get("spread_bps", np.nan) if valid else np.nan,
        "volume_5min": row.get("volume", np.nan) if valid else np.nan,
        "trade_count_5min": row.get("trade_count", np.nan) if valid else np.nan,
        "quote_updates_5min": row.get("quote_updates", np.nan) if valid else np.nan,
        "kr_close_known": row.get("kr_close_known", np.nan) if valid else np.nan,
        "usdkrw_anchor": row.get("fx_mid_asof", np.nan) if valid else np.nan,
        "parity_spread_anchor": row.get("parity_spread", np.nan) if valid else np.nan,
        "selection_version": spec.version,
    }


def select_anchors(us_5min: pd.DataFrame, spec: AnchorSpec | None = None) -> pd.DataFrame:
    """Select earliest two-bar-qualified dynamic anchor and fixed 09:30 anchor."""

    spec = spec or AnchorSpec()
    spec.validate()
    required = {
        "date_us", "bar_start_utc", "bar_end_utc", "volume", "trade_count",
        "quote_updates", "spread_bps", "quote_coverage", "quote_age_seconds",
        "valid_price", "valid_quote",
    }
    missing = required.difference(us_5min.columns)
    if missing:
        raise ValueError(f"us_5min is missing anchor columns: {sorted(missing)}")
    frame = us_5min.copy()
    frame["bar_start_utc"] = pd.to_datetime(frame["bar_start_utc"], utc=True)
    frame["bar_end_utc"] = pd.to_datetime(frame["bar_end_utc"], utc=True)
    frame["start_et"] = frame["bar_start_utc"].dt.tz_convert(NY_TZ).dt.time
    frame["end_et"] = frame["bar_end_utc"].dt.tz_convert(NY_TZ).dt.time
    frame["liquidity_score"] = liquidity_score(frame)
    frame["qualifies"] = (
        frame["valid_price"].fillna(False)
        & frame["valid_quote"].fillna(False)
        & frame["trade_count"].ge(spec.min_trades)
        & frame["volume"].ge(spec.min_volume)
        & frame["quote_updates"].ge(spec.min_quote_updates)
        & frame["spread_bps"].le(spec.max_spread_bps)
        & frame["quote_coverage"].ge(spec.min_quote_coverage)
        & frame["quote_age_seconds"].le(spec.max_quote_age_seconds)
        & frame["liquidity_score"].ge(spec.min_liquidity_score)
    )

    records: list[dict[str, object]] = []
    for date_us, day in frame.groupby("date_us", sort=True):
        day = day.sort_values("bar_end_utc").copy()
        first_completed_end = (
            datetime.combine(datetime.min.date(), spec.start_et)
            + timedelta(minutes=spec.bar_minutes)
        ).time()
        candidates = day[
            day["end_et"].ge(first_completed_end)
            & day["end_et"].le(spec.cutoff_et)
        ].copy()
        qualifying_run = candidates["qualifies"].rolling(
            spec.consecutive_bars, min_periods=spec.consecutive_bars
        ).sum().eq(spec.consecutive_bars)
        expected_span = pd.Timedelta(
            minutes=spec.bar_minutes * (spec.consecutive_bars - 1)
        )
        actual_span = candidates["bar_end_utc"] - candidates["bar_end_utc"].shift(
            spec.consecutive_bars - 1
        )
        confirmed = qualifying_run & actual_span.eq(expected_span)
        selected = candidates.loc[confirmed].head(1)
        if selected.empty:
            if candidates.empty:
                status = "premarket_bars_missing"
            elif candidates["quote_updates"].fillna(0).eq(0).all():
                status = "quotes_missing"
            elif candidates["trade_count"].fillna(0).eq(0).all():
                status = "trades_missing"
            else:
                status = "no_qualifying_anchor"
            records.append(_anchor_record(date_us, "premarket_dynamic", None, status, spec))
        else:
            row = selected.iloc[0].copy()
            row["anchor_ts_utc"] = row["bar_end_utc"]
            records.append(_anchor_record(date_us, "premarket_dynamic", row, "selected", spec))

        fixed = day[day["start_et"].eq(spec.fixed_open_et)].head(1)
        if fixed.empty:
            records.append(_anchor_record(date_us, "regular_0930", None, "fixed_open_unavailable", spec))
        else:
            row = fixed.iloc[0].copy()
            fixed_is_current = bool(
                row.get("valid_price", False)
                and row.get("valid_quote", False)
                and row.get("quote_age_seconds", np.inf) <= spec.max_quote_age_seconds
            )
            if not fixed_is_current:
                records.append(_anchor_record(date_us, "regular_0930", None, "fixed_open_stale_or_invalid", spec))
            else:
                row["anchor_ts_utc"] = row["bar_start_utc"]
                row["anchor_trade_usd"] = row.get("open", np.nan)
                records.append(_anchor_record(date_us, "regular_0930", row, "selected", spec))
    return pd.DataFrame.from_records(records)
