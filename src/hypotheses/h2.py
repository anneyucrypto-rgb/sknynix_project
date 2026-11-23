"""Thirty-minute H2 convergence models beginning at 09:30 America/New_York.

H2 measures how a thirty-minute SKHY pricing deviation closes relative to both a
static Korean-close/FX equilibrium and a prior-trained dynamic U.S.-information
reference.  It supports a fixed 09:30 clock, a two-bar liquidity-qualified
anchor, ECM speed estimates, threshold ECMs, and right-censored convergence
episodes with Kaplan--Meier summaries.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant

from src.config import PROJECT_ROOT, feature_output_path
from src.features.h1_h2 import H2_FEATURE_PATH
from src.features.utils import NY, RAW_DATA, SEOUL, _parquet_tree
from src.hypotheses.h1 import expanding_h1_parameters


ReferenceName = Literal["static", "dynamic"]
AnchorName = Literal["fixed", "liquid"]
SessionName = Literal["regular"]


@dataclass(frozen=True)
class H2Config:
    """Point-in-time construction and convergence settings for H2."""

    start_clock: str = "09:30"
    regular_open: str = "09:30"
    regular_close: str = "16:00"
    interval_minutes: int = 30
    min_h1_train: int = 40
    max_h1_p: int = 3
    max_h1_q: int = 3
    min_dynamic_days: int = 20
    min_dynamic_observations: int = 200
    min_dollar_volume: float = 150_000.0
    min_quote_updates: int = 30
    max_spread_bps: float = 75.0
    liquidity_confirmation_bars: int = 2
    convergence_band_bps: float = 25.0
    transaction_cost_bps: float = 8.0
    convergence_confirmation_bars: int = 2
    fx_max_age_minutes: int = 15

    def validate(self) -> None:
        if self.interval_minutes != 30:
            raise ValueError("H2 currently requires thirty-minute intervals")
        if self.min_h1_train < 20:
            raise ValueError("min_h1_train must be at least 20")
        if self.min_dynamic_days < 1 or self.min_dynamic_observations < 1:
            raise ValueError("dynamic-reference training thresholds must be positive")
        if self.min_dollar_volume <= 0 or self.min_quote_updates < 1:
            raise ValueError("liquidity thresholds must be positive")
        if self.max_spread_bps <= 0 or self.convergence_band_bps <= 0:
            raise ValueError("spread and convergence thresholds must be positive")
        if self.liquidity_confirmation_bars < 2:
            raise ValueError("liquidity confirmation requires at least two bars")
        if self.convergence_confirmation_bars < 2:
            raise ValueError("convergence confirmation requires at least two observations")


DYNAMIC_CONTROLS = (
    "soxx_return_from_anchor",
    "qqq_return_from_anchor",
    "news_cumulative",
)

ECM_CONTROLS = (
    "soxx_return_30min",
    "qqq_return_30min",
    "fx_return_30min",
    "news_flow",
    "log_dollar_volume",
    "spread_bps",
)


def _utc(values: object) -> pd.Series:
    return pd.to_datetime(values, utc=True).astype("datetime64[ns, UTC]")


def _local_minutes(values: pd.Series) -> pd.Series:
    local = _utc(values).dt.tz_convert(NY)
    return local.dt.hour * 60 + local.dt.minute


def _regular_grid(dates: Iterable[object], config: H2Config) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for value in sorted(pd.to_datetime(pd.Series(list(dates))).dt.date.unique()):
        start = pd.Timestamp(f"{value} {config.start_clock}", tz=NY)
        end = pd.Timestamp(f"{value} {config.regular_close}", tz=NY) - pd.Timedelta(
            minutes=config.interval_minutes
        )
        local = pd.date_range(start, end, freq=f"{config.interval_minutes}min")
        rows.append(pd.DataFrame({
            "date_us": value,
            "bar_start_utc": local.tz_convert("UTC"),
        }))
    if not rows:
        return pd.DataFrame(columns=["date_us", "bar_start_utc"])
    grid = pd.concat(rows, ignore_index=True)
    grid["bar_start_utc"] = _utc(grid["bar_start_utc"])
    grid["observation_ts_utc"] = grid["bar_start_utc"] + pd.Timedelta(
        minutes=config.interval_minutes
    )
    grid["observation_ts_utc"] = _utc(grid["observation_ts_utc"])
    grid["session"] = "regular"
    return grid


def _quote_buckets(raw: Path, config: H2Config) -> pd.DataFrame:
    quotes = _parquet_tree(raw / "us" / "market" / "SKHY_quotes")[[
        "sip_datetime", "bid_price", "ask_price",
    ]].copy()
    quotes["sip_datetime"] = _utc(quotes["sip_datetime"])
    quotes["bid_price"] = pd.to_numeric(quotes["bid_price"], errors="coerce")
    quotes["ask_price"] = pd.to_numeric(quotes["ask_price"], errors="coerce")
    quotes = quotes.loc[
        quotes["bid_price"].gt(0)
        & quotes["ask_price"].ge(quotes["bid_price"])
    ].copy()
    minute = _local_minutes(quotes["sip_datetime"])
    start_minute = int(config.start_clock[:2]) * 60 + int(config.start_clock[3:])
    close_minute = int(config.regular_close[:2]) * 60 + int(config.regular_close[3:])
    quotes = quotes.loc[minute.between(start_minute, close_minute - 1)].copy()
    quotes["bar_start_utc"] = quotes["sip_datetime"].dt.floor(
        f"{config.interval_minutes}min"
    )
    quotes = quotes.sort_values("sip_datetime")
    counts = quotes.groupby("bar_start_utc", sort=True).size().rename("quote_updates")
    last = quotes.groupby("bar_start_utc", sort=True).tail(1).set_index("bar_start_utc")
    output = last[["sip_datetime", "bid_price", "ask_price"]].join(counts).reset_index()
    output = output.rename(columns={"sip_datetime": "last_quote_ts_utc"})
    output["skhy_mid"] = (output["bid_price"] + output["ask_price"]) / 2.0
    output["spread_bps"] = (
        (output["ask_price"] - output["bid_price"]) / output["skhy_mid"] * 10_000.0
    )
    return output


def _interval_bars(raw: Path, ticker: str, config: H2Config) -> pd.DataFrame:
    """Aggregate source five-minute bars into completed H2 intervals."""
    frame = _parquet_tree(raw / "us" / "market" / f"{ticker}_5min").copy()
    frame["source_start_utc"] = _utc(frame["timestamp"])
    frame["bar_start_utc"] = frame["source_start_utc"].dt.floor(
        f"{config.interval_minutes}min"
    )
    for column in ("open", "close", "volume", "vwap", "transactions"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.sort_values("source_start_utc")
    frame["vwap_notional"] = frame["vwap"] * frame["volume"]
    grouped = frame.groupby("bar_start_utc", as_index=False, sort=True)
    output = grouped.agg(
        open=("open", "first"),
        close=("close", "last"),
        volume=("volume", "sum"),
        transactions=("transactions", "sum"),
        vwap_notional=("vwap_notional", "sum"),
    )
    output["vwap"] = output["vwap_notional"] / output["volume"].replace(0.0, np.nan)
    return output.drop(columns="vwap_notional").rename(columns={
        "open": f"{ticker.lower()}_open",
        "close": f"{ticker.lower()}_close",
        "volume": f"{ticker.lower()}_volume",
        "vwap": f"{ticker.lower()}_vwap",
        "transactions": f"{ticker.lower()}_transactions",
    })


def _market_returns(panel: pd.DataFrame, ticker: str) -> pd.DataFrame:
    close = f"{ticker.lower()}_close"
    open_column = f"{ticker.lower()}_open"
    output = panel.copy()
    output[f"{ticker.lower()}_return_30min"] = output.groupby("date_us")[close].transform(
        lambda values: np.log(values).diff()
    )
    first_bar = output.groupby("date_us")["bar_start_utc"].transform("min")
    output["reference_open"] = output[open_column].where(
        output["bar_start_utc"].eq(first_bar)
    ).groupby(output["date_us"]).transform("first")
    output[f"{ticker.lower()}_return_from_anchor"] = np.log(
        output[close] / output["reference_open"]
    )
    return output.drop(columns="reference_open")


def _attach_fx(panel: pd.DataFrame, raw: Path, config: H2Config) -> pd.DataFrame:
    fx = _parquet_tree(raw / "fx" / "USDKRW_1min")[["timestamp", "close"]].copy()
    fx["fx_available_utc"] = _utc(fx["timestamp"]) + pd.Timedelta(minutes=1)
    fx = fx.rename(columns={"close": "usdkrw"}).sort_values("fx_available_utc")
    output = pd.merge_asof(
        panel.sort_values("observation_ts_utc"),
        fx[["fx_available_utc", "usdkrw"]],
        left_on="observation_ts_utc",
        right_on="fx_available_utc",
        direction="backward",
        tolerance=pd.Timedelta(minutes=config.fx_max_age_minutes),
    )
    output["fx_return_30min"] = output.groupby("date_us")["usdkrw"].transform(
        lambda values: np.log(values).diff()
    )
    return output.sort_values(["date_us", "bar_start_utc"]).reset_index(drop=True)


def _attach_korean_close(panel: pd.DataFrame, raw: Path) -> pd.DataFrame:
    korea = _parquet_tree(raw / "korea" / "market" / "000660_daily").copy()
    korea["kr_session_date"] = pd.to_datetime(korea["date"]).dt.date
    korea["kr_available_utc"] = _utc(
        pd.to_datetime(korea["kr_session_date"].astype(str) + " 15:30")
        .dt.tz_localize(SEOUL)
    )
    price = korea["adjusted_close"] if "adjusted_close" in korea else korea["close"]
    korea["kr_close_known"] = pd.to_numeric(price, errors="coerce")
    return pd.merge_asof(
        panel.sort_values("observation_ts_utc"),
        korea[["kr_session_date", "kr_available_utc", "kr_close_known"]].sort_values(
            "kr_available_utc"
        ),
        left_on="observation_ts_utc",
        right_on="kr_available_utc",
        direction="backward",
    ).sort_values(["date_us", "bar_start_utc"]).reset_index(drop=True)


def _attach_news(panel: pd.DataFrame, raw: Path, config: H2Config) -> pd.DataFrame:
    news = _parquet_tree(raw / "us" / "news" / "SKHY_sentiment_5min").copy()
    news["bar_start_utc"] = _utc(news["timestamp"]).dt.floor(
        f"{config.interval_minutes}min"
    )
    news = news.groupby("bar_start_utc", as_index=False)[
        ["sentiment_sum", "article_count"]
    ].sum().rename(
        columns={"sentiment_sum": "news_flow", "article_count": "news_count"}
    )
    output = panel.merge(news, on="bar_start_utc", how="left")
    output[["news_flow", "news_count"]] = output[["news_flow", "news_count"]].fillna(0.0)
    output["news_cumulative"] = output.groupby("date_us")["news_flow"].cumsum()
    return output


def _h1_parameters_by_date(
    panel: pd.DataFrame,
    daily: pd.DataFrame,
    config: H2Config,
) -> pd.DataFrame:
    return expanding_h1_parameters(
        panel["date_us"].unique(),
        daily,
        min_train=config.min_h1_train,
        max_p=config.max_h1_p,
        max_q=config.max_h1_q,
        include_current_date=False,
    )


def _dynamic_parameters_by_date(
    panel: pd.DataFrame,
    config: H2Config,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    dates = sorted(panel["date_us"].unique())
    for date in dates:
        history = panel.loc[pd.to_datetime(panel["date_us"]).lt(pd.Timestamp(date))].copy()
        days = history["date_us"].nunique()
        required = ["log_skhy_mid", "static_equilibrium", *DYNAMIC_CONTROLS]
        sample = history.replace([np.inf, -np.inf], np.nan).dropna(subset=required)
        if days < config.min_dynamic_days or len(sample) < config.min_dynamic_observations:
            continue
        target = sample["log_skhy_mid"] - sample["static_equilibrium"]
        varying = [
            column for column in DYNAMIC_CONTROLS
            if float(sample[column].std(ddof=0)) > 1e-12
        ]
        if not varying:
            continue
        fit = OLS(target.astype(float), sample[varying].astype(float)).fit(
            cov_type="HAC", cov_kwds={"maxlags": 6}
        )
        record: dict[str, object] = {
            "date_us": date,
            "dynamic_training_days": days,
            "dynamic_training_n": int(fit.nobs),
            "dynamic_training_through": history["date_us"].max(),
        }
        for control in DYNAMIC_CONTROLS:
            record[f"dynamic_beta__{control}"] = (
                float(fit.params[control]) if control in fit.params else 0.0
            )
        rows.append(record)
    columns = [
        "date_us", "dynamic_training_days", "dynamic_training_n",
        "dynamic_training_through",
        *(f"dynamic_beta__{control}" for control in DYNAMIC_CONTROLS),
    ]
    return pd.DataFrame(rows, columns=columns)


def _liquidity_anchors(panel: pd.DataFrame, config: H2Config) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for date, one in panel.groupby("date_us"):
        one = one.sort_values("bar_start_utc")
        qualified = one["liquidity_score"].ge(1.0).fillna(False).to_numpy()
        timestamps = pd.to_datetime(one["bar_start_utc"], utc=True).to_numpy()
        run = 0
        anchor = pd.NaT
        previous = None
        for index, is_qualified in enumerate(qualified):
            current = pd.Timestamp(timestamps[index])
            consecutive = previous is not None and current - previous == pd.Timedelta(
                minutes=config.interval_minutes
            )
            run = run + 1 if is_qualified and (run == 0 or consecutive) else int(is_qualified)
            if run >= config.liquidity_confirmation_bars:
                anchor = current + pd.Timedelta(minutes=config.interval_minutes)
                break
            previous = current
        rows.append({"date_us": date, "liquidity_anchor_ts_utc": anchor})
    return pd.DataFrame(rows)


def build_intraday_panel(
    raw: Path = RAW_DATA,
    daily: pd.DataFrame | None = None,
    config: H2Config = H2Config(),
) -> pd.DataFrame:
    """Build the point-in-time thirty-minute H2 panel and both equilibrium gaps."""
    config.validate()
    daily = (
        pd.read_parquet(H2_FEATURE_PATH)
        if daily is None and H2_FEATURE_PATH.is_file()
        else daily
    )
    if daily is None:
        raise FileNotFoundError(f"Daily H1 panel not found: {H2_FEATURE_PATH}")

    skhy = _interval_bars(raw, "SKHY", config)
    minute = _local_minutes(skhy["bar_start_utc"])
    start_minute = int(config.start_clock[:2]) * 60 + int(config.start_clock[3:])
    close_minute = int(config.regular_close[:2]) * 60 + int(config.regular_close[3:])
    skhy = skhy.loc[minute.between(start_minute, close_minute - config.interval_minutes)]
    local = skhy["bar_start_utc"].dt.tz_convert(NY)
    skhy["date_us"] = local.dt.date
    panel = _regular_grid(skhy["date_us"].unique(), config)
    panel = panel.merge(skhy.drop(columns="date_us"), on="bar_start_utc", how="left")
    panel = panel.merge(_quote_buckets(raw, config), on="bar_start_utc", how="left")
    panel["dollar_volume"] = (
        pd.to_numeric(panel["skhy_volume"], errors="coerce")
        * pd.to_numeric(panel["skhy_vwap"], errors="coerce")
    )
    panel["log_dollar_volume"] = np.log1p(panel["dollar_volume"])

    for ticker in ("SOXX", "QQQ"):
        panel = panel.merge(_interval_bars(raw, ticker, config), on="bar_start_utc", how="left")
        panel = _market_returns(panel, ticker)
    panel = _attach_fx(panel, raw, config)
    panel = _attach_korean_close(panel, raw)
    panel = _attach_news(panel, raw, config)

    h1 = _h1_parameters_by_date(panel, daily, config)
    panel = panel.merge(h1, on="date_us", how="left")
    panel["log_skhy_mid"] = np.log(panel["skhy_mid"].where(panel["skhy_mid"].gt(0)))
    panel["k_known"] = np.log(
        (0.1 * panel["kr_close_known"] / panel["usdkrw"]).where(
            panel["kr_close_known"].gt(0) & panel["usdkrw"].gt(0)
        )
    )
    panel["static_equilibrium"] = panel["h1_intercept"] + panel["h1_theta"] * panel["k_known"]
    panel["g_static"] = panel["log_skhy_mid"] - panel["static_equilibrium"]

    dynamic = _dynamic_parameters_by_date(panel, config)
    panel = panel.merge(dynamic, on="date_us", how="left")
    dynamic_adjustment = pd.Series(0.0, index=panel.index)
    for control in DYNAMIC_CONTROLS:
        beta = f"dynamic_beta__{control}"
        dynamic_adjustment = dynamic_adjustment + panel[beta] * panel[control]
    panel["dynamic_equilibrium"] = panel["static_equilibrium"] + dynamic_adjustment
    panel["g_dynamic"] = (
        panel["log_skhy_mid"] - panel["dynamic_equilibrium"]
    ).where(panel["dynamic_training_n"].notna())

    volume_ratio = panel["dollar_volume"] / config.min_dollar_volume
    update_ratio = panel["quote_updates"] / config.min_quote_updates
    spread_ratio = config.max_spread_bps / panel["spread_bps"].replace(0.0, np.nan)
    panel["liquidity_score"] = pd.concat(
        [volume_ratio, update_ratio, spread_ratio], axis=1
    ).min(axis=1, skipna=False)
    panel["liquidity_qualified"] = panel["liquidity_score"].ge(1.0)
    panel = panel.merge(_liquidity_anchors(panel, config), on="date_us", how="left")
    panel["fixed_clock_ts_utc"] = _utc(
        pd.to_datetime(panel["date_us"].astype(str) + f" {config.start_clock}")
        .dt.tz_localize(NY)
    )
    panel["valid_fixed_observation"] = panel[[
        "skhy_mid", "usdkrw", "kr_close_known", "h1_intercept", "h1_theta",
    ]].notna().all(axis=1)
    panel["convergence_band"] = np.maximum(
        config.convergence_band_bps,
        panel["spread_bps"].fillna(config.convergence_band_bps)
        + config.transaction_cost_bps,
    ) / 10_000.0
    return panel.sort_values(["date_us", "bar_start_utc"]).reset_index(drop=True)


def _deviation_column(reference: ReferenceName) -> str:
    if reference not in {"static", "dynamic"}:
        raise ValueError("reference must be 'static' or 'dynamic'")
    return f"g_{reference}"


def _anchor_filter(frame: pd.DataFrame, anchor: AnchorName) -> pd.Series:
    if anchor == "fixed":
        return frame["observation_ts_utc"].ge(frame["fixed_clock_ts_utc"])
    if anchor == "liquid":
        return (
            frame["liquidity_anchor_ts_utc"].notna()
            & frame["observation_ts_utc"].ge(frame["liquidity_anchor_ts_utc"])
        )
    raise ValueError("anchor must be 'fixed' or 'liquid'")


def _speed(lambda_30: float) -> tuple[float, float]:
    if not -1.0 < lambda_30 < 0.0:
        return math.nan, math.nan
    remaining = 1.0 + lambda_30
    return (
        30.0 * math.log(0.5) / math.log(remaining),
        30.0 * math.log(0.1) / math.log(remaining),
    )


def fit_intraday_ecm(
    panel: pd.DataFrame,
    *,
    reference: ReferenceName = "static",
    anchor: AnchorName = "fixed",
    session: SessionName = "regular",
    controls: Iterable[str] = ECM_CONTROLS,
    hac_lags: int = 2,
) -> tuple[dict[str, float | int | str | bool], object]:
    """Estimate Delta g[t+1] on g[t] and point-in-time thirty-minute controls."""
    deviation = _deviation_column(reference)
    control_names = tuple(dict.fromkeys(controls))
    missing = set(control_names).difference(panel.columns)
    if missing:
        raise ValueError(f"Unknown H2 ECM controls: {sorted(missing)}")
    frame = panel.loc[panel["session"].eq(session) & _anchor_filter(panel, anchor)].copy()
    frame = frame.sort_values(["date_us", "observation_ts_utc"])
    grouped = frame.groupby("date_us", sort=False)
    frame["next_g"] = grouped[deviation].shift(-1)
    frame["next_ts"] = grouped["observation_ts_utc"].shift(-1)
    frame["delta_g_next"] = frame["next_g"] - frame[deviation]
    consecutive = (
        frame["next_ts"] - frame["observation_ts_utc"]
    ).eq(pd.Timedelta(minutes=30))
    required = ["delta_g_next", deviation, *control_names]
    sample = frame.loc[consecutive].replace([np.inf, -np.inf], np.nan).dropna(
        subset=required
    )
    if len(sample) <= len(control_names) + 3:
        raise ValueError("Insufficient complete thirty-minute observations for H2 ECM")
    x = add_constant(
        sample[[deviation, *control_names]].astype(float), has_constant="add"
    )
    fit = OLS(sample["delta_g_next"].astype(float), x).fit(
        cov_type="HAC", cov_kwds={"maxlags": hac_lags}
    )
    lambda_30 = float(fit.params[deviation])
    half_life, t90 = _speed(lambda_30)
    summary: dict[str, float | int | str | bool] = {
        "reference": reference,
        "anchor": anchor,
        "session": session,
        "nobs": int(fit.nobs),
        "sessions": int(sample["date_us"].nunique()),
        "controls": ", ".join(control_names),
        "lambda_30": lambda_30,
        "lambda_30_std_error": float(fit.bse[deviation]),
        "lambda_30_p_value": float(fit.pvalues[deviation]),
        "monotonic_stable": bool(-1.0 < lambda_30 < 0.0),
        "half_life_minutes": half_life,
        "t90_minutes": t90,
        "r_squared": float(fit.rsquared),
        "bic": float(fit.bic),
    }
    for term in fit.params.index:
        summary[f"coefficient__{term}"] = float(fit.params[term])
        summary[f"pvalue__{term}"] = float(fit.pvalues[term])
    return summary, fit


def fit_threshold_ecm(
    panel: pd.DataFrame,
    *,
    reference: ReferenceName = "static",
    anchor: AnchorName = "fixed",
    session: SessionName = "regular",
    controls: Iterable[str] = ECM_CONTROLS,
    hac_lags: int = 2,
) -> dict[str, float | int | str | bool]:
    """Estimate separate adjustment rates inside and outside the convergence band."""
    deviation = _deviation_column(reference)
    control_names = tuple(dict.fromkeys(controls))
    frame = panel.loc[panel["session"].eq(session) & _anchor_filter(panel, anchor)].copy()
    frame = frame.sort_values(["date_us", "observation_ts_utc"])
    grouped = frame.groupby("date_us", sort=False)
    frame["next_g"] = grouped[deviation].shift(-1)
    frame["next_ts"] = grouped["observation_ts_utc"].shift(-1)
    frame["delta_g_next"] = frame["next_g"] - frame[deviation]
    frame["g_large"] = frame[deviation].where(
        frame[deviation].abs().gt(frame["convergence_band"]), 0.0
    )
    frame["g_small"] = frame[deviation].where(
        frame[deviation].abs().le(frame["convergence_band"]), 0.0
    )
    consecutive = (
        frame["next_ts"] - frame["observation_ts_utc"]
    ).eq(pd.Timedelta(minutes=30))
    required = ["delta_g_next", "g_large", "g_small", *control_names]
    sample = frame.loc[consecutive].replace([np.inf, -np.inf], np.nan).dropna(
        subset=required
    )
    if sample["g_large"].ne(0).sum() < 5 or sample["g_small"].ne(0).sum() < 5:
        raise ValueError("Threshold H2 requires at least five observations in each regime")
    x = add_constant(
        sample[["g_large", "g_small", *control_names]].astype(float),
        has_constant="add",
    )
    fit = OLS(sample["delta_g_next"].astype(float), x).fit(
        cov_type="HAC", cov_kwds={"maxlags": hac_lags}
    )
    large = float(fit.params["g_large"])
    small = float(fit.params["g_small"])
    return {
        "reference": reference,
        "anchor": anchor,
        "session": session,
        "nobs": int(fit.nobs),
        "lambda_large": large,
        "lambda_large_p_value": float(fit.pvalues["g_large"]),
        "lambda_small": small,
        "lambda_small_p_value": float(fit.pvalues["g_small"]),
        "faster_outside_band": bool(abs(large) > abs(small)),
        "large_regime_stable": bool(-1.0 < large < 0.0),
        "small_regime_stable": bool(-1.0 < small < 0.0),
        "bic": float(fit.bic),
    }


def convergence_episodes(
    panel: pd.DataFrame,
    *,
    reference: ReferenceName = "static",
    anchor: AnchorName = "liquid",
    session: SessionName = "regular",
    confirmation_bars: int = 2,
) -> pd.DataFrame:
    """Return one convergence or right-censored episode per eligible session."""
    deviation = _deviation_column(reference)
    if confirmation_bars < 2:
        raise ValueError("convergence requires at least two consecutive observations")
    rows: list[dict[str, object]] = []
    for date, one in panel.loc[panel["session"].eq(session)].groupby("date_us"):
        one = one.sort_values("observation_ts_utc").copy()
        local_date = pd.Timestamp(date)
        clock_origin = pd.to_datetime(one["fixed_clock_ts_utc"].iloc[0], utc=True)
        cutoff = pd.Timestamp(f"{local_date.date()} 16:00", tz=NY).tz_convert("UTC")
        liquid_anchor = pd.to_datetime(one["liquidity_anchor_ts_utc"].iloc[0], utc=True)
        origin = clock_origin if anchor == "fixed" else liquid_anchor
        origin_available = pd.notna(origin)
        event_time = pd.NaT
        observed = pd.DataFrame()
        if origin_available:
            observed = one.loc[
                one["observation_ts_utc"].ge(origin)
                & one["observation_ts_utc"].le(cutoff)
            ].dropna(subset=[deviation, "convergence_band"])
        eligible = bool(origin_available and not observed.empty)
        if eligible:
            inside = observed[deviation].abs().le(observed["convergence_band"]).to_numpy()
            times = observed["observation_ts_utc"].tolist()
            run = 0
            previous = None
            for index, is_inside in enumerate(inside):
                current = pd.Timestamp(times[index])
                consecutive = previous is not None and current - previous == pd.Timedelta(minutes=30)
                run = run + 1 if is_inside and (run == 0 or consecutive) else int(is_inside)
                if run >= confirmation_bars:
                    event_time = current
                    break
                previous = current
        event = bool(pd.notna(event_time))
        end = event_time if event else cutoff
        rows.append({
            "date_us": date,
            "reference": reference,
            "anchor": anchor,
            "session": session,
            "eligible": bool(eligible),
            "liquidity_anchor_ts_utc": liquid_anchor,
            "convergence_ts_utc": event_time,
            "event_observed": event,
            "duration_clock_minutes": (
                (end - clock_origin).total_seconds() / 60.0 if eligible else math.nan
            ),
            "duration_liquid_minutes": (
                (end - liquid_anchor).total_seconds() / 60.0
                if eligible and pd.notna(liquid_anchor) else math.nan
            ),
            "right_censored": bool(eligible and not event),
        })
    return pd.DataFrame(rows)


def kaplan_meier(
    episodes: pd.DataFrame,
    *,
    duration: str = "duration_liquid_minutes",
) -> pd.DataFrame:
    """Compute a Kaplan--Meier survival curve without dropping censored sessions."""
    sample = episodes.loc[episodes["eligible"]].dropna(subset=[duration]).copy()
    if sample.empty:
        return pd.DataFrame(columns=[
            "time_minutes", "at_risk", "events", "censored", "survival",
            "convergence_probability",
        ])
    survival = 1.0
    rows: list[dict[str, float | int]] = []
    for time in sorted(sample[duration].unique()):
        at_risk = int(sample[duration].ge(time).sum())
        at_time = sample.loc[sample[duration].eq(time)]
        events = int(at_time["event_observed"].sum())
        censored = int(len(at_time) - events)
        if at_risk and events:
            survival *= 1.0 - events / at_risk
        rows.append({
            "time_minutes": float(time),
            "at_risk": at_risk,
            "events": events,
            "censored": censored,
            "survival": survival,
            "convergence_probability": 1.0 - survival,
        })
    return pd.DataFrame(rows)


def convergence_summary(
    episodes: pd.DataFrame,
    *,
    duration: str = "duration_liquid_minutes",
) -> dict[str, float | int | str]:
    curve = kaplan_meier(episodes, duration=duration)
    eligible = episodes.loc[episodes["eligible"]].dropna(subset=[duration])
    summary: dict[str, float | int | str] = {
        "duration": duration,
        "eligible_sessions": len(eligible),
        "converged_sessions": int(eligible["event_observed"].sum()),
        "no_convergence_rate": float((~eligible["event_observed"]).mean())
        if len(eligible) else math.nan,
        "median_convergence_minutes": math.nan,
    }
    if not curve.empty:
        median = curve.loc[curve["convergence_probability"].ge(0.5), "time_minutes"]
        if not median.empty:
            summary["median_convergence_minutes"] = float(median.iloc[0])
        for horizon in (30, 60, 120):
            prior = curve.loc[curve["time_minutes"].le(horizon)]
            summary[f"probability_by_{horizon}_minutes"] = (
                float(prior["convergence_probability"].iloc[-1]) if not prior.empty else 0.0
            )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--daily-input", type=Path, default=H2_FEATURE_PATH)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "features" / "experiments" / "run_id=h2-30min-convergence",
    )
    parser.add_argument("--min-h1-train", type=int, default=40)
    parser.add_argument("--min-dynamic-days", type=int, default=20)
    parser.add_argument("--min-dynamic-observations", type=int, default=200)
    parser.add_argument("--min-dollar-volume", type=float, default=150_000.0)
    parser.add_argument("--min-quote-updates", type=int, default=30)
    parser.add_argument("--max-spread-bps", type=float, default=75.0)
    parser.add_argument("--convergence-band-bps", type=float, default=25.0)
    parser.add_argument("--transaction-cost-bps", type=float, default=8.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = H2Config(
        min_h1_train=args.min_h1_train,
        min_dynamic_days=args.min_dynamic_days,
        min_dynamic_observations=args.min_dynamic_observations,
        min_dollar_volume=args.min_dollar_volume,
        min_quote_updates=args.min_quote_updates,
        max_spread_bps=args.max_spread_bps,
        convergence_band_bps=args.convergence_band_bps,
        transaction_cost_bps=args.transaction_cost_bps,
    )
    daily = pd.read_parquet(args.daily_input)
    panel = build_intraday_panel(RAW_DATA, daily, config)
    output = feature_output_path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(output / "h2_intraday_panel.parquet", index=False)
    pd.DataFrame([asdict(config)]).to_parquet(output / "h2_config.parquet", index=False)

    model_rows: list[dict[str, object]] = []
    threshold_rows: list[dict[str, object]] = []
    episode_frames: list[pd.DataFrame] = []
    curve_frames: list[pd.DataFrame] = []
    survival_rows: list[dict[str, object]] = []
    for reference in ("static", "dynamic"):
        for anchor in ("fixed", "liquid"):
            for session in ("regular",):
                try:
                    summary, _ = fit_intraday_ecm(
                        panel, reference=reference, anchor=anchor, session=session
                    )
                    model_rows.append(summary)
                except ValueError:
                    pass
                try:
                    threshold_rows.append(fit_threshold_ecm(
                        panel, reference=reference, anchor=anchor, session=session
                    ))
                except ValueError:
                    pass
                episodes = convergence_episodes(
                    panel,
                    reference=reference,
                    anchor=anchor,
                    session=session,
                    confirmation_bars=config.convergence_confirmation_bars,
                )
                episode_frames.append(episodes)
                for duration in ("duration_clock_minutes", "duration_liquid_minutes"):
                    curve = kaplan_meier(episodes, duration=duration)
                    if not curve.empty:
                        curve.insert(0, "duration", duration)
                        curve.insert(0, "session", session)
                        curve.insert(0, "anchor", anchor)
                        curve.insert(0, "reference", reference)
                        curve_frames.append(curve)
                    survival_rows.append({
                        "reference": reference,
                        "anchor": anchor,
                        "session": session,
                        **convergence_summary(episodes, duration=duration),
                    })

    pd.DataFrame(model_rows).to_parquet(output / "h2_ecm_models.parquet", index=False)
    pd.DataFrame(threshold_rows).to_parquet(output / "h2_threshold_models.parquet", index=False)
    pd.concat(episode_frames, ignore_index=True).to_parquet(
        output / "h2_convergence_episodes.parquet", index=False
    )
    (pd.concat(curve_frames, ignore_index=True) if curve_frames else pd.DataFrame()).to_parquet(
        output / "h2_kaplan_meier.parquet", index=False
    )
    pd.DataFrame(survival_rows).to_parquet(
        output / "h2_convergence_summary.parquet", index=False
    )
    print(f"[H2] Wrote {len(panel):,} thirty-minute rows to {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
