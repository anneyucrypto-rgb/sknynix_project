"""Shared dataset contracts, validation, and market feature calculations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT, feature_output_path


NY = "America/New_York"
SEOUL = "Asia/Seoul"
RAW_DATA = PROJECT_ROOT / "data" / "raw"


def write_feature_table(frame: pd.DataFrame, path: Path) -> Path:
    """Publish a complete Parquet snapshot inside the feature directory."""
    output = feature_output_path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".staging-", dir=output.parent) as staging:
        temporary = Path(staging) / output.name
        frame.to_parquet(temporary, index=False)
        temporary.replace(output)
    return output


def _parquet_tree(path: Path) -> pd.DataFrame:
    files = sorted(path.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No Parquet files under {path}")
    return pd.concat((pd.read_parquet(file) for file in files), ignore_index=True)


def _utc(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["timestamp"] = pd.to_datetime(output["timestamp"], utc=True)
    return output.sort_values("timestamp")


def _asof_market_close(
    events: pd.DataFrame,
    event_column: str,
    minute: pd.DataFrame,
    output_column: str,
    *,
    tolerance: str | None = None,
) -> pd.Series:
    """Return the last minute close available at each event timestamp."""

    left = events[[event_column]].copy()
    left["_row_id"] = np.arange(len(left))
    right = _utc(minute)[["timestamp", "close"]].copy()
    available = f"{output_column}_available_utc"
    right[available] = right["timestamp"] + pd.Timedelta(minutes=1)
    right = right.rename(columns={"close": output_column})
    merged = pd.merge_asof(
        left.sort_values(event_column),
        right[[available, output_column]].sort_values(available),
        left_on=event_column,
        right_on=available,
        direction="backward",
        tolerance=pd.Timedelta(tolerance) if tolerance else None,
    ).sort_values("_row_id")
    return merged[output_column].reset_index(drop=True)


def _index_gauge_features(
    frame: pd.DataFrame, minute: pd.DataFrame, prefix: str
) -> pd.DataFrame:
    """Attach prior-close-to-07:00 and 07:00-to-09:25 returns."""

    market = _utc(minute)
    local = market["timestamp"].dt.tz_convert(NY)
    market["date_us"] = local.dt.date
    minute_of_day = local.dt.hour * 60 + local.dt.minute
    regular = market.loc[minute_of_day.between(570, 959)].copy()
    closes = regular.groupby("date_us", as_index=False).tail(1)[
        ["timestamp", "close"]
    ]
    previous_ts = f"{prefix}_previous_close_ts"
    previous_price = f"{prefix}_previous_close"
    closes[previous_ts] = closes["timestamp"] + pd.Timedelta(minutes=1)
    closes = closes.rename(columns={"close": previous_price})

    output = frame.copy().reset_index(drop=True)
    previous = pd.merge_asof(
        output[["forecast_cutoff_utc"]]
        .assign(_row_id=np.arange(len(output)))
        .sort_values("forecast_cutoff_utc"),
        closes[[previous_ts, previous_price]].sort_values(previous_ts),
        left_on="forecast_cutoff_utc",
        right_on=previous_ts,
        direction="backward",
        allow_exact_matches=False,
    ).sort_values("_row_id")
    price_0700 = _asof_market_close(
        output,
        "premarket_start_utc",
        market,
        f"{prefix}_price_0700",
        tolerance="5min",
    )
    price_cutoff = _asof_market_close(
        output,
        "forecast_cutoff_utc",
        market,
        f"{prefix}_price_cutoff",
        tolerance="5min",
    )
    output[f"{prefix}_extended_return"] = np.log(
        price_0700.to_numpy() / previous[previous_price].to_numpy()
    )
    output[f"{prefix}_premarket_return"] = np.log(
        price_cutoff.to_numpy() / price_0700.to_numpy()
    )
    return output


def build_overnight_features(raw: Path = RAW_DATA) -> pd.DataFrame:
    """Build one point-in-time forecast and execution row per US session."""

    skhy = _utc(_parquet_tree(raw / "us" / "market" / "SKHY_1min"))
    soxx = _utc(_parquet_tree(raw / "us" / "market" / "SOXX_1min"))

    local = skhy["timestamp"].dt.tz_convert(NY)
    skhy["date_us"] = local.dt.date
    minute = local.dt.hour * 60 + local.dt.minute
    opens = skhy.loc[minute.eq(570), ["date_us", "timestamp", "open"]].copy()
    opens = opens.rename(
        columns={"timestamp": "open_0930_ts_utc", "open": "open_0930"}
    )
    opens["forecast_cutoff_utc"] = (
        pd.to_datetime(opens["date_us"].astype(str) + " 09:25")
        .dt.tz_localize(NY)
        .dt.tz_convert("UTC")
    )
    opens["premarket_start_utc"] = (
        pd.to_datetime(opens["date_us"].astype(str) + " 07:00")
        .dt.tz_localize(NY)
        .dt.tz_convert("UTC")
    )

    regular = skhy.loc[minute.between(570, 959)].copy()
    closes = regular.groupby("date_us", as_index=False).tail(1)[
        ["date_us", "timestamp", "close"]
    ].copy()
    closes["previous_close_ts_utc"] = closes["timestamp"] + pd.Timedelta(
        minutes=1
    )
    closes = closes.rename(columns={"close": "previous_us_close"})
    frame = pd.merge_asof(
        opens.sort_values("forecast_cutoff_utc"),
        closes[["previous_close_ts_utc", "previous_us_close"]].sort_values(
            "previous_close_ts_utc"
        ),
        left_on="forecast_cutoff_utc",
        right_on="previous_close_ts_utc",
        direction="backward",
        allow_exact_matches=False,
    )
    frame["overnight_return"] = np.log(
        frame["open_0930"] / frame["previous_us_close"]
    )

    entry_bars = skhy.loc[minute.eq(565), ["date_us", "open"]].copy()
    entry_bars = entry_bars.drop_duplicates("date_us", keep="last").rename(
        columns={"open": "entry_0925"}
    )
    frame = frame.merge(entry_bars, on="date_us", how="left")
    frame["tradable_return_0925_0930"] = np.log(
        frame["open_0930"] / frame["entry_0925"]
    )

    execution = skhy.loc[minute.between(565, 569)].copy()
    execution["dollar_volume"] = (
        pd.to_numeric(execution["volume"], errors="coerce").fillna(0.0)
        * pd.to_numeric(execution["vwap"], errors="coerce").fillna(
            pd.to_numeric(execution["close"], errors="coerce")
        )
    )
    execution_summary = (
        execution.groupby("date_us", as_index=False)["dollar_volume"]
        .sum()
        .rename(columns={"dollar_volume": "premarket_exit_dollar_volume"})
    )
    frame = frame.merge(execution_summary, on="date_us", how="left")

    opening = skhy.loc[
        minute.eq(570), ["date_us", "volume", "vwap", "close"]
    ].copy()
    opening["opening_dollar_volume"] = (
        pd.to_numeric(opening["volume"], errors="coerce").fillna(0.0)
        * pd.to_numeric(opening["vwap"], errors="coerce").fillna(
            pd.to_numeric(opening["close"], errors="coerce")
        )
    )
    frame = frame.merge(
        opening[["date_us", "opening_dollar_volume"]].drop_duplicates(
            "date_us", keep="last"
        ),
        on="date_us",
        how="left",
    )
    for offset, minute_value in enumerate(range(565, 570)):
        path_bar = skhy.loc[
            minute.eq(minute_value), ["date_us", "high", "low"]
        ].copy()
        path_bar = path_bar.drop_duplicates("date_us", keep="last").rename(
            columns={
                "high": f"trade_high_m{offset}",
                "low": f"trade_low_m{offset}",
            }
        )
        frame = frame.merge(path_bar, on="date_us", how="left")

    kr = _parquet_tree(raw / "korea" / "market" / "000660_daily").copy()
    kr["kr_session_date"] = pd.to_datetime(kr["date"]).dt.date
    kr["kr_available_at_utc"] = (
        pd.to_datetime(kr["kr_session_date"].astype(str) + " 15:30")
        .dt.tz_localize(SEOUL)
        .dt.tz_convert("UTC")
    )
    kr["kr_day_return"] = np.log(
        pd.to_numeric(kr["close"], errors="coerce")
        / pd.to_numeric(kr["open"], errors="coerce")
    )
    kr["kr_close_known"] = pd.to_numeric(
        kr.get("adjusted_close", kr["close"]), errors="coerce"
    )
    frame = pd.merge_asof(
        frame.sort_values("forecast_cutoff_utc"),
        kr[
            [
                "kr_session_date",
                "kr_available_at_utc",
                "kr_day_return",
                "kr_close_known",
            ]
        ].sort_values("kr_available_at_utc"),
        left_on="forecast_cutoff_utc",
        right_on="kr_available_at_utc",
        direction="backward",
    )

    frame = _index_gauge_features(frame, soxx, "soxx")
    return frame.sort_values("date_us").reset_index(drop=True)


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
    "h1": DatasetContract(
        "h1",
        "one eligible daily observation",
        ("date_us",),
        (
            "date_us", "a_d", "k_d", "delta_a", "delta_k", "a_l1",
            "k_l1", "sample_eligible",
        ),
        description="Daily ARDL--UECM inputs for the H1 level relationship",
    ),
    "h2": DatasetContract(
        "h2",
        "one U.S. date by fixed thirty-minute response window",
        ("date_us", "anchor_type", "horizon_min"),
        (
            "date_us", "anchor_type", "anchor_ts_utc", "horizon_min",
            "endpoint_ts_utc", "ect_anchor", "liquidity_score",
            "ect_x_liquidity", "spread_bps", "kr_return_known", "adr_return_h",
            "soxx_return_h", "usdkrw_return_h", "ect_change_h",
            "crosses_open", "valid_outcome", "h1_model_id",
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


class DatasetValidationError(ValueError):
    pass


def validate_contract(
    frame: pd.DataFrame,
    contract: str | DatasetContract,
    *,
    allow_empty: bool = False,
) -> None:
    spec = CONTRACTS[contract] if isinstance(contract, str) else contract
    missing = set(spec.required_columns).difference(frame.columns)
    if missing:
        raise DatasetValidationError(
            f"{spec.name} is missing required columns: {sorted(missing)}"
        )
    if frame.empty and not allow_empty:
        raise DatasetValidationError(f"{spec.name} contains no observations")
    duplicated = frame.duplicated(list(spec.primary_key), keep=False)
    if duplicated.any():
        raise DatasetValidationError(
            f"{spec.name} has {int(duplicated.sum())} duplicate primary-key rows"
        )
    for column in spec.timestamp_columns:
        values = pd.to_datetime(frame[column], errors="coerce", utc=True)
        if frame[column].notna().any() and values.notna().sum() != frame[column].notna().sum():
            raise DatasetValidationError(f"{spec.name}.{column} has invalid timestamps")


def validate_no_lookahead(frame: pd.DataFrame, dataset: str) -> None:
    violations: list[str] = []
    if dataset == "daily":
        mask = frame["kr_available_at_utc"].notna() & (
            pd.to_datetime(frame["kr_available_at_utc"], utc=True)
            > pd.to_datetime(frame["us_close_ts_utc"], utc=True)
        )
        if mask.any():
            violations.append(f"{int(mask.sum())} Korean closes unavailable at U.S. close")
        fx = frame["fx_asof_ts_utc"].notna() & (
            pd.to_datetime(frame["fx_asof_ts_utc"], utc=True)
            > pd.to_datetime(frame["us_close_ts_utc"], utc=True)
        )
        if fx.any():
            violations.append(f"{int(fx.sum())} future FX observations")
    elif dataset == "us_5min":
        cutoff = pd.to_datetime(frame["bar_end_utc"], utc=True)
        for column in ("last_quote_ts", "fx_ts", "available"):
            if column in frame:
                observed = pd.to_datetime(frame[column], utc=True)
                bad = observed.notna() & (observed > cutoff)
                if bad.any():
                    violations.append(f"{int(bad.sum())} future {column} observations")
    elif dataset == "h2":
        bad = frame["valid_outcome"].fillna(False) & (
            pd.to_datetime(frame["endpoint_ts_utc"], utc=True)
            <= pd.to_datetime(frame["anchor_ts_utc"], utc=True)
        )
        if bad.any():
            violations.append(f"{int(bad.sum())} non-future H2 endpoints")
        if "outcome_ts" in frame:
            outcome = pd.to_datetime(frame["outcome_ts"], utc=True)
            endpoint = pd.to_datetime(frame["endpoint_ts_utc"], utc=True)
            future = frame["valid_outcome"].fillna(False) & (outcome > endpoint)
            stale = frame["valid_outcome"].fillna(False) & ((endpoint - outcome) > pd.Timedelta("60s"))
            if future.any() or stale.any():
                violations.append(f"{int((future | stale).sum())} invalid H2 endpoint matches")
    elif dataset == "h3":
        if frame["information_cutoff_utc"].isna().any():
            violations.append("H3 contains missing information cutoffs")
        us_to_kr = frame["direction"].eq("us_to_korea")
        if us_to_kr.any():
            target_open = pd.to_datetime(
                frame.loc[us_to_kr, "target_session_date"].astype(str) + " 09:00"
            ).dt.tz_localize("Asia/Seoul").dt.tz_convert("UTC")
            cutoff = pd.to_datetime(
                frame.loc[us_to_kr, "information_cutoff_utc"], utc=True
            )
            bad = cutoff >= target_open
            if bad.any():
                violations.append(f"{int(bad.sum())} U.S.-to-Korea predictors miss the target-session cutoff")
    if violations:
        raise DatasetValidationError(f"{dataset} look-ahead validation failed: " + "; ".join(violations))


def validation_summary(frame: pd.DataFrame) -> dict[str, object]:
    return {
        "rows": len(frame),
        "columns": len(frame.columns),
        "missing_fraction": float(frame.isna().mean().mean()) if len(frame) else 1.0,
        "duplicate_rows": int(frame.duplicated().sum()),
    }


DEFAULT_ORDINARY_SHARES_PER_ADS = 0.1


def theoretical_adr_usd(
    kr_price_krw: float | pd.Series,
    usdkrw: float | pd.Series,
    ordinary_shares_per_ads: float = DEFAULT_ORDINARY_SHARES_PER_ADS,
) -> float | pd.Series:
    """Return the theoretical USD ADS price from Korean shares and FX."""

    if ordinary_shares_per_ads <= 0:
        raise ValueError("ordinary_shares_per_ads must be positive")
    return kr_price_krw * ordinary_shares_per_ads / usdkrw


def log_kr_usd(
    kr_price_krw: float | pd.Series,
    usdkrw: float | pd.Series,
    ordinary_shares_per_ads: float = DEFAULT_ORDINARY_SHARES_PER_ADS,
) -> float | pd.Series:
    theoretical = theoretical_adr_usd(
        kr_price_krw, usdkrw, ordinary_shares_per_ads
    )
    return np.log(theoretical)


def equilibrium_error(
    log_adr: float | pd.Series,
    log_korean_usd: float | pd.Series,
    *,
    intercept: float = 0.0,
    theta: float = 1.0,
) -> float | pd.Series:
    return log_adr - intercept - theta * log_korean_usd


def add_parity_columns(
    frame: pd.DataFrame,
    *,
    adr_column: str,
    kr_column: str,
    fx_column: str,
    ordinary_shares_per_ads: float = DEFAULT_ORDINARY_SHARES_PER_ADS,
) -> pd.DataFrame:
    """Add theoretical price, logs, and parity spread with invalids as null."""

    output = frame.copy()
    valid = (
        output[adr_column].gt(0)
        & output[kr_column].gt(0)
        & output[fx_column].gt(0)
    )
    theoretical = theoretical_adr_usd(
        output[kr_column], output[fx_column], ordinary_shares_per_ads
    )
    output["theoretical_adr_usd"] = theoretical.where(valid)
    output["log_adr"] = np.log(output[adr_column]).where(valid)
    output["log_kr_usd"] = np.log(theoretical).where(valid)
    output["parity_spread"] = (
        output["log_adr"] - output["log_kr_usd"]
    ).where(valid)
    return output
