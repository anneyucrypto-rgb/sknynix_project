"""Materialize the point-in-time daily panel used to estimate H1 and H2.

The builder creates the close-to-open target, FX-adjusted Korean predictor,
level variables, and daily control candidates. H1 is estimated by
``src.hypotheses.h1``; H2 and nested model selection are estimated by
``src.hypotheses.overnight_ablation``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT
from src.features.utils import NY, RAW_DATA, SEOUL, _parquet_tree, _utc, write_feature_table


H1_FEATURE_PATH = PROJECT_ROOT / "data" / "features" / "daily" / "part-00000.parquet"
H2_FEATURE_PATH = PROJECT_ROOT / "data" / "features" / "overnight_h2" / "part-00000.parquet"
PREDICTOR = "kr_fx_adjusted_return"

# Ablate sources as coherent information sets rather than cherry-picking fields.
CONTROL_GROUPS: dict[str, tuple[str, ...]] = {
    "soxx": ("soxx_daily_return",),
    "sentiment": ("skhy_sentiment_daily_normalized",),
    "korea_semiconductor": ("korea_semiconductor_daily_return",),
    "kosdaq": ("kosdaq_daily_return",),
    "kospi": ("kospi_daily_return",),
    "kr_market_proxies": (
        "kr_proxy_flkr_return", "kr_proxy_usdkrw_return",
    ),
    "us_inflation": ("us_inflation_yoy", "us_core_inflation_change"),
    "us_funding_conditions": (
        "funding_effr_change", "funding_sofr_effr_spread",
        "funding_obfr_effr_spread", "funding_cp_spread",
        "funding_rrp_log_change",
    ),
    "vix": ("vix_daily_return",),
    "us_treasury": ("treasury_2y_change", "treasury_10y_change"),
}


def _utc_ns(values: object) -> pd.Series:
    """Normalize pandas/Arrow timestamp units before as-of joins."""
    return pd.to_datetime(values, utc=True).astype("datetime64[ns, UTC]")


def _price_column(frame: pd.DataFrame) -> pd.Series:
    return pd.to_numeric(
        frame["adjusted_close"] if "adjusted_close" in frame else frame["close"],
        errors="coerce",
    )


def _overnight_target(raw: Path) -> pd.DataFrame:
    bars = _utc(_parquet_tree(raw / "us" / "market" / "SKHY_1min"))
    local = bars["timestamp"].dt.tz_convert(NY)
    minute = local.dt.hour * 60 + local.dt.minute
    regular = bars.loc[minute.between(570, 959)].copy()
    regular["date_us"] = local.loc[regular.index].dt.date

    opens = regular.loc[minute.loc[regular.index].eq(570), ["date_us", "timestamp", "open"]]
    opens = opens.drop_duplicates("date_us", keep="first").rename(
        columns={"timestamp": "us_open_utc", "open": "open_0930"}
    )
    closes = regular.groupby("date_us", as_index=False).tail(1)[["timestamp", "close"]].copy()
    closes["close_available_utc"] = closes["timestamp"] + pd.Timedelta(minutes=1)
    closes = closes.rename(columns={"close": "previous_us_close"})
    panel = pd.merge_asof(
        opens.sort_values("us_open_utc"),
        closes[["close_available_utc", "previous_us_close"]].sort_values("close_available_utc"),
        left_on="us_open_utc", right_on="close_available_utc",
        direction="backward", allow_exact_matches=False,
    )
    panel["overnight_return"] = np.log(panel["open_0930"] / panel["previous_us_close"])
    return panel


def _korean_predictor(raw: Path) -> pd.DataFrame:
    ordinary = _parquet_tree(raw / "korea" / "market" / "000660_daily").copy()
    ordinary["kr_session_date"] = pd.to_datetime(ordinary["date"]).dt.date
    ordinary["kr_available_utc"] = (
        pd.to_datetime(ordinary["kr_session_date"].astype(str) + " 15:30")
        .dt.tz_localize(SEOUL).dt.tz_convert("UTC")
    )
    ordinary["kr_available_utc"] = _utc_ns(ordinary["kr_available_utc"])
    fx = _utc(_parquet_tree(raw / "fx" / "USDKRW_1min"))[["timestamp", "close"]].copy()
    fx["fx_available_utc"] = fx["timestamp"] + pd.Timedelta(minutes=1)
    fx["fx_available_utc"] = _utc_ns(fx["fx_available_utc"])
    fx = fx.rename(columns={"close": "usdkrw_at_kr_close"})
    ordinary = pd.merge_asof(
        ordinary.sort_values("kr_available_utc"),
        fx[["fx_available_utc", "usdkrw_at_kr_close"]].sort_values("fx_available_utc"),
        left_on="kr_available_utc", right_on="fx_available_utc",
        direction="backward", tolerance=pd.Timedelta("3D"),
    )
    ordinary["kr_fx_adjusted_close_usd"] = (
        0.1 * _price_column(ordinary) / ordinary["usdkrw_at_kr_close"]
    )
    ordinary[PREDICTOR] = np.log(ordinary["kr_fx_adjusted_close_usd"]).diff()
    return ordinary[["kr_session_date", "kr_available_utc", "kr_fx_adjusted_close_usd", PREDICTOR]]


def _korean_return(raw: Path, dataset: str, column: str) -> pd.DataFrame:
    frame = _parquet_tree(raw / "korea" / "market" / "sector_daily" / dataset).copy()
    frame["kr_session_date"] = pd.to_datetime(frame["date"]).dt.date
    frame[column] = np.log(_price_column(frame)).diff()
    return frame[["kr_session_date", column]].drop_duplicates("kr_session_date", keep="last")


def _us_regular_return(raw: Path) -> pd.DataFrame:
    bars = _utc(_parquet_tree(raw / "us" / "market" / "SOXX_1min"))
    local = bars["timestamp"].dt.tz_convert(NY)
    minute = local.dt.hour * 60 + local.dt.minute
    regular = bars.loc[minute.between(570, 959)].copy()
    regular["date"] = local.loc[regular.index].dt.date
    close = regular.groupby("date", as_index=False).tail(1)[["timestamp", "close"]].copy()
    close["available_utc"] = close["timestamp"] + pd.Timedelta(minutes=1)
    close["soxx_daily_return"] = np.log(close["close"]).diff()
    return close[["available_utc", "soxx_daily_return"]]


def _sentiment(raw: Path, start: object, end: object) -> pd.DataFrame:
    events = _parquet_tree(raw / "us" / "news" / "SKHY_sentiment_events").copy()
    events["published_utc"] = pd.to_datetime(events["published_utc"], utc=True)
    events["date"] = events["published_utc"].dt.tz_convert(NY).dt.normalize().dt.tz_localize(None)
    counts = events.assign(
        positive=events["sentiment_score"].eq(1).astype(int),
        negative=events["sentiment_score"].eq(-1).astype(int),
    ).groupby("date")[["positive", "negative"]].sum()
    calendar = pd.DataFrame({"date": pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq="D")})
    daily = calendar.merge(counts, left_on="date", right_index=True, how="left").fillna(0.0)
    log_odds = np.log((daily["positive"] + 1.0) / (daily["negative"] + 1.0))
    daily["skhy_sentiment_daily_normalized"] = np.tanh(log_odds / 2.0)
    daily["available_utc"] = (
        (daily["date"] + pd.Timedelta(days=1)).dt.tz_localize(NY).dt.tz_convert("UTC")
    )
    return daily[["available_utc", "skhy_sentiment_daily_normalized"]]


def _merge_asof(panel: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    panel = panel.copy()
    right = right.copy()
    panel["us_open_utc"] = _utc_ns(panel["us_open_utc"])
    right["available_utc"] = _utc_ns(right["available_utc"])
    return pd.merge_asof(
        panel.sort_values("us_open_utc"), right.sort_values("available_utc"),
        left_on="us_open_utc", right_on="available_utc", direction="backward",
    ).drop(columns="available_utc")


def _daily_us_control(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    output = frame.copy().sort_values("date")
    output[columns] = output[columns].ffill()
    output["available_utc"] = pd.to_datetime(output["date"], utc=True) + pd.Timedelta(days=1)
    return output[["available_utc", *columns]]


def build_h1_h2_features(raw: Path = RAW_DATA) -> pd.DataFrame:
    """Build one date-level observation for each SKHY overnight return."""

    panel = _overnight_target(raw)
    korean = _korean_predictor(raw)
    panel = pd.merge_asof(
        panel.sort_values("us_open_utc"), korean.sort_values("kr_available_utc"),
        left_on="us_open_utc", right_on="kr_available_utc", direction="backward",
        tolerance=pd.Timedelta("4D"),
    )
    for dataset, column in (
        ("KOREA_SEMICONDUCTOR_daily", "korea_semiconductor_daily_return"),
        ("KOSDAQ_daily", "kosdaq_daily_return"),
        ("KOSPI_daily", "kospi_daily_return"),
    ):
        panel = panel.merge(_korean_return(raw, dataset, column), on="kr_session_date", how="left")

    panel = _merge_asof(panel, _us_regular_return(raw))
    sentiment_start = pd.Timestamp(panel["date_us"].min()) - pd.Timedelta(days=1)
    panel = _merge_asof(panel, _sentiment(raw, sentiment_start, panel["date_us"].max()))

    proxies = _utc(_parquet_tree(raw / "korea" / "economic" / "proxies" / "KR_MARKET_PROXIES_daily"))
    proxies = proxies.sort_values(["ticker", "timestamp"])
    proxies["proxy_return"] = proxies.groupby("ticker")["close"].transform(lambda x: np.log(x).diff())
    proxy_names = {
        "EWY": "kr_proxy_ewy_return", "FLKR": "kr_proxy_flkr_return",
        "C:USDKRW": "kr_proxy_usdkrw_return",
    }
    for ticker, column in proxy_names.items():
        one = proxies.loc[proxies["ticker"].eq(ticker), ["timestamp", "proxy_return"]].rename(
            columns={"timestamp": "available_utc", "proxy_return": column}
        )
        one["available_utc"] = one["available_utc"] + pd.Timedelta(days=1)
        panel = _merge_asof(panel, one)

    inflation = _parquet_tree(raw / "us" / "economic" / "inflation" / "US_INFLATION_monthly").sort_values("date")
    cpi = pd.to_numeric(inflation["cpi"], errors="coerce").ffill()
    supplied_yoy = pd.to_numeric(inflation["cpi_year_over_year"], errors="coerce")
    inflation["us_inflation_yoy"] = supplied_yoy.fillna(100.0 * cpi.pct_change(12))
    inflation["us_core_inflation_change"] = pd.to_numeric(
        inflation["cpi_core"], errors="coerce"
    ).ffill().pct_change(fill_method=None)
    # No release timestamp is supplied. The 15th of the next month is an
    # explicit, conservative availability convention.
    inflation["available_utc"] = (
        pd.to_datetime(inflation["date"]) + pd.offsets.MonthBegin(1) + pd.Timedelta(days=14)
    ).dt.tz_localize("UTC")
    panel = _merge_asof(panel, inflation[["available_utc", "us_inflation_yoy", "us_core_inflation_change"]])

    funding = _parquet_tree(raw / "us" / "economic" / "policy" / "US_FUNDING_CONDITIONS_daily").sort_values("date")
    effr = pd.to_numeric(funding["effective_fed_funds_rate"], errors="coerce")
    funding["funding_effr_change"] = effr.diff()
    funding["funding_sofr_effr_spread"] = pd.to_numeric(funding["secured_overnight_financing_rate"], errors="coerce") - effr
    funding["funding_obfr_effr_spread"] = pd.to_numeric(funding["overnight_bank_funding_rate"], errors="coerce") - effr
    funding["funding_cp_spread"] = pd.to_numeric(funding["financial_commercial_paper_90d_rate"], errors="coerce") - pd.to_numeric(funding["nonfinancial_commercial_paper_90d_rate"], errors="coerce")
    funding["funding_rrp_log_change"] = np.log1p(pd.to_numeric(funding["fed_overnight_reverse_repo_treasury_amount"], errors="coerce")).diff()
    panel = _merge_asof(panel, _daily_us_control(funding, list(CONTROL_GROUPS["us_funding_conditions"])))

    vix = _parquet_tree(raw / "us" / "economic" / "risk" / "VIX_daily").sort_values("date")
    vix["vix_daily_return"] = np.log(_price_column(vix)).diff()
    panel = _merge_asof(panel, _daily_us_control(vix, ["vix_daily_return"]))

    treasury = _parquet_tree(raw / "us" / "economic" / "treasury" / "US_TREASURY_daily").sort_values("date")
    treasury["treasury_2y_change"] = pd.to_numeric(treasury["yield_2_year"], errors="coerce").diff()
    treasury["treasury_10y_change"] = pd.to_numeric(treasury["yield_10_year"], errors="coerce").diff()
    panel = _merge_asof(panel, _daily_us_control(treasury, list(CONTROL_GROUPS["us_treasury"])))

    panel = panel.sort_values("date_us").reset_index(drop=True)
    panel["valid_h1"] = panel[["overnight_return", PREDICTOR]].notna().all(axis=1)
    all_controls = [column for columns in CONTROL_GROUPS.values() for column in columns]
    panel["available_control_count"] = panel[all_controls].notna().sum(axis=1)
    # Timestamps enforce point-in-time joins but are not model variables.
    return panel.drop(
        columns=["us_open_utc", "close_available_utc", "kr_available_utc", "fx_available_utc"],
        errors="ignore",
    )


def build_h2_features() -> pd.DataFrame:
    """Backward-compatible name for the unified daily H1/H2 panel."""
    return build_h1_h2_features()


def write_h2_features() -> pd.DataFrame:
    frame = build_h1_h2_features()
    write_feature_table(frame, H1_FEATURE_PATH)
    output = write_feature_table(frame, H2_FEATURE_PATH)
    print(f"Wrote {len(frame)} daily H1/H2 rows to {output}")
    return frame


def main() -> None:
    write_h2_features()


if __name__ == "__main__":
    main()
