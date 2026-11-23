"""H3: test whether the next Korean session completes price convergence.

The dependent variable is the next observed SK hynix Korean-share return after
the U.S. close. The key predictor is the causal H1 equilibrium error measured
at that U.S. close, defined as SKHY minus the FX-adjusted Korean reference.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant

from src.config import PROJECT_ROOT, feature_output_path
from src.features.h1_h2 import H1_FEATURE_PATH
from src.features.utils import NY, RAW_DATA, SEOUL, _parquet_tree
from src.hypotheses.h1 import expanding_h1_parameters


DEFAULT_CONTROLS = (
    "soxx_return",
    "fx_return",
    "log_skhy_dollar_volume",
)


@dataclass(frozen=True)
class H3Config:
    """Point-in-time construction and inference settings for H3."""

    min_h1_train: int = 40
    max_h1_p: int = 3
    max_h1_q: int = 3
    fx_max_age_minutes: int = 30
    hac_lags: int = 5
    controls: tuple[str, ...] = DEFAULT_CONTROLS

    def validate(self) -> None:
        if self.min_h1_train < 20:
            raise ValueError("min_h1_train must be at least 20")
        if self.max_h1_p < 0 or self.max_h1_q < 0:
            raise ValueError("H1 lag limits must be non-negative")
        if self.fx_max_age_minutes < 1 or self.hac_lags < 0:
            raise ValueError("FX age must be positive and HAC lags non-negative")
        unknown = set(self.controls).difference(DEFAULT_CONTROLS)
        if unknown:
            raise ValueError(f"Unknown H3 controls: {sorted(unknown)}")


def _utc(values: object) -> pd.Series:
    return pd.to_datetime(values, utc=True).astype("datetime64[ns, UTC]")


def _price(frame: pd.DataFrame) -> pd.Series:
    column = "adjusted_close" if "adjusted_close" in frame else "close"
    return pd.to_numeric(frame[column], errors="coerce")


def _us_regular_sessions(raw: Path, ticker: str) -> pd.DataFrame:
    bars = _parquet_tree(raw / "us" / "market" / f"{ticker}_5min").copy()
    bars["bar_start_utc"] = _utc(bars["timestamp"])
    local = bars["bar_start_utc"].dt.tz_convert(NY)
    minute = local.dt.hour * 60 + local.dt.minute
    bars = bars.loc[minute.between(570, 955)].copy()
    bars["date_us"] = local.loc[bars.index].dt.date
    for column in ("open", "close", "volume", "vwap"):
        bars[column] = pd.to_numeric(bars[column], errors="coerce")
    bars = bars.sort_values("bar_start_utc")
    bars["bar_dollar_volume"] = bars["volume"] * bars["vwap"].fillna(bars["close"])

    close_rows = bars.groupby("date_us", as_index=False).tail(1)[
        ["date_us", "bar_start_utc", "close"]
    ].copy()
    close_rows["us_close_utc"] = close_rows["bar_start_utc"] + pd.Timedelta(minutes=5)
    close_rows = close_rows.rename(columns={"close": f"{ticker.lower()}_close"})
    volume = bars.groupby("date_us", as_index=False)["bar_dollar_volume"].sum(
        min_count=1
    )
    output = close_rows.merge(volume, on="date_us", how="left")
    if ticker != "SKHY":
        output = output.drop(columns="bar_dollar_volume")
    return output.drop(columns="bar_start_utc").sort_values("date_us")


def _attach_fx(panel: pd.DataFrame, raw: Path, config: H3Config) -> pd.DataFrame:
    fx = _parquet_tree(raw / "fx" / "USDKRW_1min")[["timestamp", "close"]].copy()
    fx["fx_available_utc"] = _utc(fx["timestamp"]) + pd.Timedelta(minutes=1)
    fx["usdkrw_us_close"] = pd.to_numeric(fx["close"], errors="coerce")
    output = pd.merge_asof(
        panel.sort_values("us_close_utc"),
        fx[["fx_available_utc", "usdkrw_us_close"]].sort_values("fx_available_utc"),
        left_on="us_close_utc",
        right_on="fx_available_utc",
        direction="backward",
        tolerance=pd.Timedelta(minutes=config.fx_max_age_minutes),
    )
    output = output.sort_values("date_us")
    output["fx_return"] = np.log(output["usdkrw_us_close"]).diff()
    return output


def _korean_sessions(raw: Path) -> pd.DataFrame:
    frame = _parquet_tree(raw / "korea" / "market" / "000660_daily").copy()
    frame["kr_session_date"] = pd.to_datetime(frame["date"]).dt.date
    frame["kr_available_utc"] = _utc(
        pd.to_datetime(frame["kr_session_date"].astype(str) + " 15:30")
        .dt.tz_localize(SEOUL)
    )
    frame["kr_close"] = _price(frame)
    return frame[["kr_session_date", "kr_available_utc", "kr_close"]].sort_values(
        "kr_available_utc"
    )


def pair_next_korean_session(
    us_panel: pd.DataFrame,
    korean: pd.DataFrame,
) -> pd.DataFrame:
    """Attach the last known and first strictly future Korean sessions."""
    left = us_panel.copy().sort_values("us_close_utc")
    right = korean.copy().sort_values("kr_available_utc")
    current = right.rename(columns={
        "kr_session_date": "kr_reference_date",
        "kr_available_utc": "kr_reference_available_utc",
        "kr_close": "kr_reference_close",
    })
    output = pd.merge_asof(
        left,
        current,
        left_on="us_close_utc",
        right_on="kr_reference_available_utc",
        direction="backward",
    )
    future = right.rename(columns={
        "kr_session_date": "kr_target_date",
        "kr_available_utc": "kr_target_available_utc",
        "kr_close": "kr_target_close",
    })
    output = pd.merge_asof(
        output.sort_values("us_close_utc"),
        future,
        left_on="us_close_utc",
        right_on="kr_target_available_utc",
        direction="forward",
        allow_exact_matches=False,
    )
    output["kr_next_return"] = np.log(
        output["kr_target_close"] / output["kr_reference_close"]
    )
    last_source = output.groupby("kr_target_date")["us_close_utc"].transform("max")
    output["last_us_close_before_target"] = output["us_close_utc"].eq(last_source)
    return output


def build_h3_panel(
    raw: Path = RAW_DATA,
    daily: pd.DataFrame | None = None,
    config: H3Config = H3Config(),
) -> pd.DataFrame:
    """Construct one causal U.S.-close to next-Korean-session observation."""
    config.validate()
    if daily is None:
        if not H1_FEATURE_PATH.is_file():
            raise FileNotFoundError(f"Daily H1 panel not found: {H1_FEATURE_PATH}")
        daily = pd.read_parquet(H1_FEATURE_PATH)

    skhy = _us_regular_sessions(raw, "SKHY").rename(
        columns={"skhy_close": "adr_close"}
    )
    soxx = _us_regular_sessions(raw, "SOXX")[["date_us", "soxx_close"]]
    panel = skhy.merge(soxx, on="date_us", how="left").sort_values("date_us")
    panel["adr_return"] = np.log(panel["adr_close"]).diff()
    panel["soxx_return"] = np.log(panel["soxx_close"]).diff()
    panel["log_skhy_dollar_volume"] = np.log1p(panel["bar_dollar_volume"])
    panel = _attach_fx(panel, raw, config)
    panel = pair_next_korean_session(panel, _korean_sessions(raw))

    parameters = expanding_h1_parameters(
        panel["date_us"].unique(),
        daily,
        min_train=config.min_h1_train,
        max_p=config.max_h1_p,
        max_q=config.max_h1_q,
        include_current_date=True,
    )
    panel = panel.merge(parameters, on="date_us", how="left")
    panel["log_adr_us_close"] = np.log(panel["adr_close"].where(panel["adr_close"].gt(0)))
    panel["log_kr_reference_usd"] = np.log(
        (0.1 * panel["kr_reference_close"] / panel["usdkrw_us_close"]).where(
            panel["kr_reference_close"].gt(0) & panel["usdkrw_us_close"].gt(0)
        )
    )
    panel["ect_us_close"] = (
        panel["log_adr_us_close"]
        - panel["h1_intercept"]
        - panel["h1_theta"] * panel["log_kr_reference_usd"]
    )
    required = ["kr_next_return", "ect_us_close", "adr_return", *config.controls]
    panel["valid_h3"] = (
        panel[required].replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
        & panel["last_us_close_before_target"].fillna(False)
    )
    panel["information_cutoff_utc"] = panel["us_close_utc"]
    return panel.sort_values("date_us").reset_index(drop=True)


def fit_h3_model(
    panel: pd.DataFrame,
    *,
    controls: Iterable[str] = DEFAULT_CONTROLS,
    hac_lags: int = 5,
) -> tuple[dict[str, float | int | str | bool], pd.DataFrame, object]:
    """Estimate the next-Korean-session convergence regression with HAC SEs."""
    control_names = tuple(dict.fromkeys(controls))
    regressors = ("ect_us_close", "adr_return", *control_names)
    missing = set(("kr_next_return", *regressors)).difference(panel.columns)
    if missing:
        raise ValueError(f"H3 panel is missing columns: {sorted(missing)}")
    candidate = panel.loc[panel["valid_h3"].fillna(False)] if "valid_h3" in panel else panel
    sample = candidate.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["kr_next_return", *regressors]
    )
    if len(sample) <= len(regressors) + 2:
        raise ValueError("Insufficient complete observations for H3")
    fit = OLS(
        sample["kr_next_return"].astype(float),
        add_constant(sample[list(regressors)].astype(float), has_constant="add"),
    ).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})
    lambda_k = float(fit.params["ect_us_close"])
    lambda_se = float(fit.bse["ect_us_close"])
    t_value = lambda_k / lambda_se if lambda_se > 0 else math.nan
    positive_p = 0.5 * math.erfc(t_value / math.sqrt(2.0)) if np.isfinite(t_value) else math.nan
    summary: dict[str, float | int | str | bool] = {
        "nobs": int(fit.nobs),
        "source_sessions": int(sample["date_us"].nunique()),
        "target_sessions": int(sample["kr_target_date"].nunique())
        if "kr_target_date" in sample else int(fit.nobs),
        "controls": ", ".join(control_names),
        "lambda_k": lambda_k,
        "lambda_k_std_error": lambda_se,
        "lambda_k_t_value": t_value,
        "lambda_k_p_value_two_sided": float(fit.pvalues["ect_us_close"]),
        "lambda_k_p_value_positive": positive_p,
        "expected_positive_sign": bool(lambda_k > 0.0),
        "positive_at_5pct": bool(lambda_k > 0.0 and positive_p < 0.05),
        "adr_return_coefficient": float(fit.params["adr_return"]),
        "adr_return_p_value": float(fit.pvalues["adr_return"]),
        "r_squared": float(fit.rsquared),
        "adjusted_r_squared": float(fit.rsquared_adj),
        "bic": float(fit.bic),
    }
    coefficients = pd.DataFrame({
        "term": fit.params.index,
        "coefficient": fit.params.to_numpy(dtype=float),
        "std_error_hac": fit.bse.to_numpy(dtype=float),
        "t_value": fit.tvalues.to_numpy(dtype=float),
        "p_value_two_sided": fit.pvalues.to_numpy(dtype=float),
    })
    return summary, coefficients, fit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--daily-input", type=Path, default=H1_FEATURE_PATH)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "features" / "experiments" / "run_id=h3-next-korean-session",
    )
    parser.add_argument("--min-h1-train", type=int, default=40)
    parser.add_argument("--max-h1-p", type=int, default=3)
    parser.add_argument("--max-h1-q", type=int, default=3)
    parser.add_argument("--hac-lags", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = H3Config(
        min_h1_train=args.min_h1_train,
        max_h1_p=args.max_h1_p,
        max_h1_q=args.max_h1_q,
        hac_lags=args.hac_lags,
    )
    daily = pd.read_parquet(args.daily_input)
    panel = build_h3_panel(RAW_DATA, daily, config)
    summary, coefficients, _ = fit_h3_model(
        panel, controls=config.controls, hac_lags=config.hac_lags
    )
    output = feature_output_path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(output / "h3_panel.parquet", index=False)
    pd.DataFrame([asdict(config)]).to_parquet(output / "h3_config.parquet", index=False)
    pd.DataFrame([summary]).to_parquet(output / "h3_model.parquet", index=False)
    coefficients.to_parquet(output / "h3_coefficients.parquet", index=False)
    print(f"[H3] Wrote {len(panel):,} paired rows to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
