"""Fit the retained bivariate-H1/SOXX-gauge H2 overnight model."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.ardl import ARDL, UECM

from src.config import PROJECT_ROOT


NY = "America/New_York"
SEOUL = "Asia/Seoul"

H1_CONTROL_SETS: dict[str, tuple[str, ...]] = {"bivariate": ()}


@dataclass(frozen=True)
class OvernightSpec:
    static_controls: tuple[str, ...]
    include_ect: bool


H2_SPECS: dict[str, OvernightSpec] = {
    "korean_ect_soxx_gauges": OvernightSpec(
        (
            "kr_day_return",
            "soxx_extended_return",
            "soxx_premarket_return",
        ),
        True,
    )
}


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


def build_overnight_dataset(max_q: int = 0) -> pd.DataFrame:
    """Build one retained-model row per U.S. trading session."""

    del max_q  # q is fixed at zero in the retained specification.
    raw = PROJECT_ROOT / "data" / "raw"
    skhy = _utc(_parquet_tree(raw / "us" / "market" / "SKHY_1min"))
    soxx = _utc(_parquet_tree(raw / "us" / "market" / "SOXX_1min"))
    fx = _utc(_parquet_tree(raw / "fx" / "USDKRW_1min"))

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

    fx_available = fx[["timestamp", "close"]].copy()
    fx_available["fx_available_at_utc"] = fx_available["timestamp"] + pd.Timedelta(
        minutes=1
    )
    fx_available = fx_available.rename(columns={"close": "usdkrw_cutoff"})
    frame = pd.merge_asof(
        frame.sort_values("forecast_cutoff_utc"),
        fx_available[["fx_available_at_utc", "usdkrw_cutoff"]].sort_values(
            "fx_available_at_utc"
        ),
        left_on="forecast_cutoff_utc",
        right_on="fx_available_at_utc",
        direction="backward",
        tolerance=pd.Timedelta("15min"),
    )
    frame["log_kr_usd_cutoff"] = np.log(
        0.1 * frame["kr_close_known"] / frame["usdkrw_cutoff"]
    )
    frame = _index_gauge_features(frame, soxx, "soxx")

    frame = frame.sort_values("date_us").reset_index(drop=True)
    for lag in range(1, 4):
        frame[f"overnight_return_l{lag}"] = frame["overnight_return"].shift(lag)
    frame["valid_overnight"] = frame[
        [
            "overnight_return",
            "kr_day_return",
            "previous_us_close",
            "open_0930",
            "usdkrw_cutoff",
            "log_kr_usd_cutoff",
            "soxx_extended_return",
            "soxx_premarket_return",
            "tradable_return_0925_0930",
        ]
    ].notna().all(axis=1)
    return frame


def fit_h1(
    frame: pd.DataFrame, controls: tuple[str, ...], max_lags: int
) -> tuple[dict[str, object], object]:
    """Select a UECM-convertible bivariate ARDL by BIC."""

    required = ["log_adr", "log_kr_usd", *controls]
    sample = frame[required].replace([np.inf, -np.inf], np.nan).dropna().copy()
    fixed = sample[list(controls)] if controls else None
    candidates: list[tuple[float, int, int, ARDL]] = []
    for p_order in range(1, max_lags + 1):
        for q_order in range(1, max_lags + 1):
            model = ARDL(
                sample["log_adr"],
                lags=p_order,
                exog=sample[["log_kr_usd"]],
                order=q_order,
                fixed=fixed,
                trend="c",
                missing="drop",
            )
            candidates.append((float(model.fit().bic), p_order, q_order, model))
    _, p_order, q_order, selected = min(candidates, key=lambda item: item[0])
    model = UECM.from_ardl(selected)
    hac_lags = max(1, int(math.floor(4 * (len(sample) / 100) ** (2 / 9))))
    result = model.fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})
    bounds = result.bounds_test(
        case=3, cov_type="HAC", cov_kwds={"maxlags": hac_lags}
    )
    rho = float(result.params["log_adr.L1"])
    record: dict[str, object] = {
        "nobs": int(result.nobs),
        "controls": ", ".join(controls) if controls else "none",
        "ardl_order": f"({p_order}, {q_order})",
        "bic": float(result.bic),
        "h1_lambda": rho,
        "h1_lambda_p": float(result.pvalues["log_adr.L1"]),
        "bounds_f": float(bounds.stat),
        "bounds_upper_p": float(bounds.p_values["upper"]),
        "intercept": -float(result.params["const"]) / rho,
        "theta": -float(result.params["log_kr_usd.L1"]) / rho,
    }
    return record, result


def _design(
    frame: pd.DataFrame, spec: OvernightSpec, p_order: int, q_order: int
) -> pd.DataFrame:
    del q_order
    columns = [*spec.static_controls]
    if spec.include_ect:
        columns.append("ect")
    columns.extend(f"overnight_return_l{lag}" for lag in range(1, p_order + 1))
    return sm.add_constant(frame[columns].astype(float), has_constant="add")


def _required(spec: OvernightSpec, p_order: int, q_order: int) -> list[str]:
    del q_order
    columns = ["overnight_return", *spec.static_controls]
    if spec.include_ect:
        columns.append("ect")
    columns.extend(f"overnight_return_l{lag}" for lag in range(1, p_order + 1))
    return columns


def select_orders(
    frame: pd.DataFrame, spec: OvernightSpec, max_p: int, max_q: int
) -> tuple[float, int, int]:
    """Select p by BIC; q is fixed at zero for aggregate SOXX gauges."""

    del max_q
    common = frame.replace([np.inf, -np.inf], np.nan).dropna(
        subset=_required(spec, max_p, 0)
    )
    candidates: list[tuple[float, int, int]] = []
    for p_order in range(max_p + 1):
        fit = sm.OLS(
            common["overnight_return"], _design(common, spec, p_order, 0)
        ).fit()
        candidates.append((float(fit.bic), p_order, 0))
    return min(candidates)


def _joint_control_pvalue(fit: object, spec: OvernightSpec) -> float:
    terms = [column for column in spec.static_controls if column != "kr_day_return"]
    restriction = np.zeros((len(terms), len(fit.params)))
    for row, term in enumerate(terms):
        restriction[row, fit.params.index.get_loc(term)] = 1.0
    try:
        test = fit.wald_test(restriction, scalar=True)
        return float(np.asarray(test.pvalue).item())
    except (ValueError, np.linalg.LinAlgError):
        return math.nan


def walk_forward(
    frame: pd.DataFrame,
    spec: OvernightSpec,
    p_order: int,
    q_order: int,
    min_train: int,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    """Generate expanding-window forecasts using earlier dates only."""

    sample = (
        frame.replace([np.inf, -np.inf], np.nan)
        .dropna(subset=["date_us", *_required(spec, p_order, q_order)])
        .sort_values("date_us")
        .reset_index(drop=True)
    )
    if len(sample) <= min_train:
        raise ValueError(f"Need more than {min_train} complete rows; got {len(sample)}")
    records: list[dict[str, object]] = []
    for index in range(min_train, len(sample)):
        train = sample.iloc[:index]
        test = sample.iloc[[index]]
        x_train = _design(train, spec, p_order, q_order)
        x_test = _design(test, spec, p_order, q_order).reindex(
            columns=x_train.columns, fill_value=0.0
        )
        fit = sm.OLS(train["overnight_return"], x_train).fit()
        forecast = float(fit.predict(x_test).iloc[0])
        actual = float(test["overnight_return"].iloc[0])
        record: dict[str, object] = {
            "date_us": test["date_us"].iloc[0],
            "actual_return": actual,
            "forecast_return": forecast,
            "forecast_error": actual - forecast,
            "correct_direction": np.sign(actual) == np.sign(forecast),
            "tradable_return_0925_0930": (
                float(test["tradable_return_0925_0930"].iloc[0])
                if "tradable_return_0925_0930" in test
                else math.nan
            ),
            "training_n": len(train),
            "signal_available_utc": (
                test["forecast_cutoff_utc"].iloc[0]
                if "forecast_cutoff_utc" in test
                else pd.NaT
            ),
            "overnight_entry_utc": (
                test["previous_close_ts_utc"].iloc[0]
                if "previous_close_ts_utc" in test
                else pd.NaT
            ),
        }
        for column in (
            "previous_us_close",
            "entry_0925",
            "open_0930",
            "premarket_exit_dollar_volume",
            "opening_dollar_volume",
            *(f"trade_high_m{offset}" for offset in range(5)),
            *(f"trade_low_m{offset}" for offset in range(5)),
        ):
            if column in test:
                record[column] = float(test[column].iloc[0])
        records.append(record)

    forecasts = pd.DataFrame(records)
    errors = forecasts["forecast_error"]
    actual = forecasts["actual_return"]
    zero_sse = float(np.square(actual).sum())
    model_sse = float(np.square(errors).sum())
    metrics: dict[str, float | int] = {
        "n_forecasts": len(forecasts),
        "rmse": float(np.sqrt(np.mean(np.square(errors)))),
        "mae": float(np.mean(np.abs(errors))),
        "bias": float(errors.mean()),
        "direction_accuracy": float(forecasts["correct_direction"].mean()),
        "oos_r2_zero": 1.0 - model_sse / zero_sse if zero_sse > 0 else math.nan,
    }
    return forecasts, metrics


def common_comparison_dates(
    overnight: pd.DataFrame, h2_summary: pd.DataFrame
) -> set[object]:
    """Return dates complete for the single retained H2 equation."""

    row = h2_summary.loc[h2_summary["status"].eq("ok")].iloc[0]
    spec = H2_SPECS[str(row["h2_spec"])]
    required = [
        "date_us",
        "tradable_return_0925_0930",
        *_required(spec, int(row["p_daily"]), int(row["q_five_minute"])),
    ]
    return set(
        overnight.replace([np.inf, -np.inf], np.nan)
        .dropna(subset=required)["date_us"]
    )


def run_model(
    run_id: str, max_p: int, max_q: int, min_train: int
) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    output_data = (
        PROJECT_ROOT
        / "data"
        / "features"
        / "experiments"
        / f"run_id={run_id}"
    )
    report_dir = PROJECT_ROOT / "reports" / f"run_id={run_id}"
    output_data.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    daily = pd.read_parquet(
        PROJECT_ROOT / "data" / "features" / "daily" / "part-00000.parquet"
    ).copy()
    overnight = build_overnight_dataset(max_q=0)

    print("[OVERNIGHT:H1] bivariate", flush=True)
    h1_result, _ = fit_h1(daily, (), max_lags=max_p)
    h1_summary = pd.DataFrame(
        [{"h1_spec": "bivariate", "status": "ok", "error": "", **h1_result}]
    )
    h1_row = h1_summary.iloc[0]
    overnight["ect"] = (
        np.log(overnight["previous_us_close"])
        - float(h1_row["intercept"])
        - float(h1_row["theta"]) * overnight["log_kr_usd_cutoff"]
    )
    overnight.to_parquet(output_data / "overnight.parquet", index=False)

    h2_name = "korean_ect_soxx_gauges"
    spec = H2_SPECS[h2_name]
    print(f"[OVERNIGHT:H2] H1=bivariate H2={h2_name}", flush=True)
    bic, p_order, q_order = select_orders(overnight, spec, max_p, max_q)
    estimation = overnight.replace([np.inf, -np.inf], np.nan).dropna(
        subset=_required(spec, p_order, q_order)
    )
    fit = sm.OLS(
        estimation["overnight_return"],
        _design(estimation, spec, p_order, q_order),
    ).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
    record: dict[str, object] = {
        "h1_spec": "bivariate",
        "h2_spec": h2_name,
        "status": "ok",
        "error": "",
        "bic": bic,
        "p_daily": p_order,
        "q_five_minute": q_order,
        "nobs": int(fit.nobs),
        "r_squared": float(fit.rsquared),
        "h2_controls": ", ".join(spec.static_controls),
        "added_controls_joint_p": _joint_control_pvalue(fit, spec),
        "h2_intercept": float(fit.params["const"]),
        "korean_return_coefficient": float(fit.params["kr_day_return"]),
        "korean_return_p": float(fit.pvalues["kr_day_return"]),
        "ect_coefficient": float(fit.params["ect"]),
        "ect_p": float(fit.pvalues["ect"]),
    }
    for term in fit.params.index:
        record[f"coefficient__{term}"] = float(fit.params[term])
        record[f"pvalue__{term}"] = float(fit.pvalues[term])
    h2_summary = pd.DataFrame([record])
    h2_summary["ect_q_global"] = h2_summary["ect_p"]
    h2_summary["added_controls_joint_q_global"] = h2_summary[
        "added_controls_joint_p"
    ]

    dates = common_comparison_dates(overnight, h2_summary)
    if len(dates) <= min_train:
        raise ValueError(
            f"Retained overnight sample has {len(dates)} rows; "
            f"need more than min_train={min_train}"
        )
    _, metrics = walk_forward(
        overnight.loc[overnight["date_us"].isin(dates)],
        spec,
        p_order,
        q_order,
        min_train,
    )
    h2_summary["common_comparison_n"] = len(dates)
    for column, value in metrics.items():
        h2_summary[column] = value
    h2_summary["forecast_rank"] = 1

    h1_summary.to_parquet(report_dir / "h1_model.parquet", index=False)
    h2_path = report_dir / "h2_model.parquet"
    h2_summary.to_parquet(h2_path, index=False)
    return h1_summary, h2_summary, h2_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="overnight-best")
    parser.add_argument("--max-p", type=int, default=3)
    parser.add_argument("--max-q", type=int, default=0)
    parser.add_argument("--min-train", type=int, default=40)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _, _, report = run_model(args.run_id, args.max_p, args.max_q, args.min_train)
    print(f"[OVERNIGHT] Wrote {report}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
