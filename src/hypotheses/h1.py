"""Daily ARDL--UECM estimation for the SKHY long-run equilibrium.

This module is the single implementation of H1.  It estimates the unrestricted
error-correction representation, derives the long-run intercept and slope,
constructs the equilibrium error, performs bounds inference, and measures
rolling parameter stability.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant

from src.config import PROJECT_ROOT, feature_output_path
from src.features.h1_h2 import H2_FEATURE_PATH


A_LEVEL = "a_d"
K_LEVEL = "k_d"
A_LAG = "a_l1"
K_LAG = "k_l1"
DELTA_A = "delta_a"
DELTA_K = "delta_k"


@dataclass(frozen=True)
class H1Spec:
    """Selected daily UECM lag orders and contemporaneous controls."""

    p_order: int
    q_order: int
    controls: tuple[str, ...] = ()


@dataclass
class H1Estimate:
    """An H1 estimate with the exact estimation sample retained for inference."""

    summary: dict[str, float | int | str]
    fit: object
    sample: pd.DataFrame
    prepared: pd.DataFrame
    spec: H1Spec


# Narayan (2005), Case III (unrestricted intercept, no trend), k=1.  Narayan
# reports finite-sample F bounds for selected sample sizes.  The lookup uses the
# next available sample-size row, which is conservative for an in-between n.
# Values are ordered as 10% I(0), 10% I(1), 5% I(0), 5% I(1), 1% I(0), 1% I(1).
_NARAYAN_CASE_III_K1: dict[int, tuple[float, ...]] = {
    30: (4.290, 5.080, 5.395, 6.350, 8.170, 9.285),
    35: (4.225, 5.050, 5.290, 6.175, 7.870, 8.960),
    45: (4.225, 5.020, 5.235, 6.135, 7.740, 8.650),
    55: (4.155, 4.925, 5.125, 6.045, 7.435, 8.460),
    65: (4.175, 4.930, 5.130, 5.980, 7.320, 8.435),
    75: (4.150, 4.885, 5.140, 5.920, 7.225, 8.300),
}

# Pesaran, Shin, and Smith (2001), Case III asymptotic bounds for k=1.
_PSS_CASE_III_K1 = (4.040, 4.780, 4.940, 5.730, 6.840, 7.840)


def _levels(frame: pd.DataFrame) -> pd.DataFrame:
    """Return dated log price levels from either supported daily panel schema."""
    output = frame.copy()
    if {"previous_us_close", "kr_fx_adjusted_close_usd"}.issubset(output):
        us = pd.to_numeric(output["previous_us_close"], errors="coerce")
        kr = pd.to_numeric(output["kr_fx_adjusted_close_usd"], errors="coerce")
        output[A_LEVEL] = np.log(us.where(us.gt(0)))
        output[K_LEVEL] = np.log(kr.where(kr.gt(0)))
    elif {"log_adr", "log_kr_usd"}.issubset(output):
        output[A_LEVEL] = pd.to_numeric(output["log_adr"], errors="coerce")
        output[K_LEVEL] = pd.to_numeric(output["log_kr_usd"], errors="coerce")
    elif not {A_LEVEL, K_LEVEL}.issubset(output):
        raise ValueError(
            "H1 requires previous_us_close/kr_fx_adjusted_close_usd, "
            "log_adr/log_kr_usd, or precomputed a_d/k_d columns"
        )
    if "date_us" not in output:
        raise ValueError("H1 requires date_us")
    return output.sort_values("date_us").reset_index(drop=True)


def prepare_uecm_data(
    frame: pd.DataFrame,
    *,
    max_p: int = 3,
    max_q: int = 3,
    controls: Iterable[str] = (),
) -> pd.DataFrame:
    """Create the level and difference terms used by every candidate UECM."""
    if max_p < 0 or max_q < 0:
        raise ValueError("max_p and max_q must be non-negative")
    output = _levels(frame)
    control_names = tuple(dict.fromkeys(controls))
    missing = set(control_names).difference(output.columns)
    if missing:
        raise ValueError(f"H1 controls are missing from the daily panel: {sorted(missing)}")

    output[DELTA_A] = output[A_LEVEL].diff()
    output[DELTA_K] = output[K_LEVEL].diff()
    output[A_LAG] = output[A_LEVEL].shift(1)
    output[K_LAG] = output[K_LEVEL].shift(1)
    for lag in range(1, max_p + 1):
        output[f"{DELTA_A}_l{lag}"] = output[DELTA_A].shift(lag)
    for lag in range(1, max_q + 1):
        output[f"{DELTA_K}_l{lag}"] = output[DELTA_K].shift(lag)
    return output


def _regressors(spec: H1Spec) -> tuple[str, ...]:
    columns = [A_LAG, K_LAG, DELTA_K]
    columns.extend(f"{DELTA_A}_l{lag}" for lag in range(1, spec.p_order + 1))
    columns.extend(f"{DELTA_K}_l{lag}" for lag in range(1, spec.q_order + 1))
    columns.extend(spec.controls)
    return tuple(columns)


def _fit_order(
    prepared: pd.DataFrame,
    spec: H1Spec,
    *,
    hac_lags: int | None,
) -> tuple[object, pd.DataFrame]:
    required = ["date_us", DELTA_A, *_regressors(spec)]
    sample = (
        prepared.replace([np.inf, -np.inf], np.nan)
        .dropna(subset=required)
        .sort_values("date_us")
        .reset_index(drop=True)
    )
    if sample["date_us"].duplicated().any():
        raise ValueError("H1 expects one observation per daily date")
    if len(sample) <= len(_regressors(spec)) + 2:
        raise ValueError("Insufficient complete observations for the requested H1 UECM")
    x = add_constant(sample[list(_regressors(spec))].astype(float), has_constant="add")
    model = OLS(sample[DELTA_A].astype(float), x)
    if hac_lags is None:
        return model.fit(), sample
    return model.fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags}), sample


def _long_run_summary(fit: object, spec: H1Spec, sample: pd.DataFrame) -> dict[str, float | int | str]:
    rho_a = float(fit.params[A_LAG])
    rho_k = float(fit.params[K_LAG])
    uecm_constant = float(fit.params["const"])
    if math.isclose(rho_a, 0.0, abs_tol=1e-12):
        raise ValueError("rho_a is numerically zero; the long-run equilibrium is unidentified")
    intercept = -uecm_constant / rho_a
    theta = -rho_k / rho_a

    # Delta-method uncertainty for theta = -rho_k / rho_a.
    covariance = fit.cov_params().loc[[A_LAG, K_LAG], [A_LAG, K_LAG]].to_numpy()
    gradient = np.array([rho_k / (rho_a * rho_a), -1.0 / rho_a])
    theta_se = float(np.sqrt(max(float(gradient @ covariance @ gradient), 0.0)))
    theta_z = theta / theta_se if theta_se > 0 else math.nan
    theta_p = math.erfc(abs(theta_z) / math.sqrt(2.0)) if np.isfinite(theta_z) else math.nan

    ect = sample[A_LEVEL] - intercept - theta * sample[K_LEVEL]
    if len(ect) > 2:
        persistence_fit = OLS(
            ect.iloc[1:].to_numpy(),
            add_constant(ect.iloc[:-1].to_numpy(), has_constant="add"),
        ).fit()
        residual_persistence = float(persistence_fit.params[1])
    else:
        residual_persistence = math.nan

    return {
        "model": "daily_ardl_uecm",
        "nobs": int(fit.nobs),
        "p_order": spec.p_order,
        "q_order": spec.q_order,
        "controls": ", ".join(spec.controls),
        "uecm_constant": uecm_constant,
        "rho_a": rho_a,
        "rho_k": rho_k,
        "intercept": intercept,
        "theta": theta,
        "theta_std_error": theta_se,
        "theta_p": theta_p,
        "residual_persistence": residual_persistence,
        "r_squared": float(fit.rsquared),
        "bic": float(fit.bic),
    }


def estimate_h1(
    frame: pd.DataFrame,
    controls: Iterable[str] = (),
    *,
    max_p: int = 3,
    max_q: int = 3,
    hac_lags: int = 3,
) -> H1Estimate:
    """Select daily UECM lag orders by BIC and estimate H1 with HAC inference."""
    control_names = tuple(dict.fromkeys(controls))
    prepared = prepare_uecm_data(
        frame, max_p=max_p, max_q=max_q, controls=control_names
    )
    candidates: list[tuple[float, int, int]] = []
    for p_order in range(max_p + 1):
        for q_order in range(max_q + 1):
            spec = H1Spec(p_order, q_order, control_names)
            try:
                selection_fit, _ = _fit_order(prepared, spec, hac_lags=None)
            except ValueError:
                continue
            if np.isfinite(selection_fit.bic):
                candidates.append((float(selection_fit.bic), p_order, q_order))
    if not candidates:
        raise ValueError("No estimable H1 lag-order candidate")
    _, p_order, q_order = min(candidates)
    spec = H1Spec(p_order, q_order, control_names)
    fit, sample = _fit_order(prepared, spec, hac_lags=hac_lags)
    summary = _long_run_summary(fit, spec, sample)
    return H1Estimate(summary, fit, sample, prepared, spec)


def expanding_h1_parameters(
    dates: Iterable[object],
    daily: pd.DataFrame,
    *,
    min_train: int = 40,
    max_p: int = 3,
    max_q: int = 3,
    include_current_date: bool = False,
) -> pd.DataFrame:
    """Estimate causal H1 parameters for each requested information date.

    Intraday H2 excludes the current date. H3 may include it because the daily
    row dated ``d`` contains prices already known by the U.S. close on ``d``.
    """
    daily_dates = pd.to_datetime(daily["date_us"])
    rows: list[dict[str, object]] = []
    for date in sorted(pd.to_datetime(pd.Series(list(dates))).dt.date.unique()):
        cutoff = pd.Timestamp(date)
        mask = daily_dates.le(cutoff) if include_current_date else daily_dates.lt(cutoff)
        history = daily.loc[mask].sort_values("date_us")
        if len(history) < min_train:
            continue
        try:
            estimate = estimate_h1(history, max_p=max_p, max_q=max_q)
        except ValueError:
            continue
        rows.append({
            "date_us": date,
            "h1_intercept": float(estimate.summary["intercept"]),
            "h1_theta": float(estimate.summary["theta"]),
            "h1_training_n": int(estimate.summary["nobs"]),
            "h1_training_through": history["date_us"].iloc[-1],
        })
    return pd.DataFrame(rows, columns=[
        "date_us", "h1_intercept", "h1_theta", "h1_training_n",
        "h1_training_through",
    ])


def fit_h1_equilibrium(
    frame: pd.DataFrame,
    controls: Iterable[str] = (),
    *,
    max_p: int = 3,
    max_q: int = 3,
    hac_lags: int = 3,
) -> tuple[dict[str, float | int | str], object]:
    """Compatibility wrapper returning the canonical ARDL H1 summary and fit."""
    estimate = estimate_h1(
        frame, controls, max_p=max_p, max_q=max_q, hac_lags=hac_lags
    )
    return estimate.summary, estimate.fit


def equilibrium_error(
    frame: pd.DataFrame, h1: dict[str, float | int | str]
) -> pd.Series:
    """Return ECT_d = a_d - intercept - theta*k_d in original row order."""
    if {"previous_us_close", "kr_fx_adjusted_close_usd"}.issubset(frame):
        us = pd.to_numeric(frame["previous_us_close"], errors="coerce")
        kr = pd.to_numeric(frame["kr_fx_adjusted_close_usd"], errors="coerce")
        a_level = np.log(us.where(us.gt(0)))
        k_level = np.log(kr.where(kr.gt(0)))
    elif {"log_adr", "log_kr_usd"}.issubset(frame):
        a_level = pd.to_numeric(frame["log_adr"], errors="coerce")
        k_level = pd.to_numeric(frame["log_kr_usd"], errors="coerce")
    elif {A_LEVEL, K_LEVEL}.issubset(frame):
        a_level = pd.to_numeric(frame[A_LEVEL], errors="coerce")
        k_level = pd.to_numeric(frame[K_LEVEL], errors="coerce")
    else:
        raise ValueError("Cannot construct H1 equilibrium error from this frame")
    return (
        a_level
        - float(h1["intercept"])
        - float(h1["theta"]) * k_level
    )


def narayan_case_iii_bounds(nobs: int) -> dict[str, float | int | str] | None:
    """Return applicable Narayan Case-III finite-sample F bounds for k=1."""
    if nobs < 30 or nobs > 75:
        return None
    table_n = next(size for size in _NARAYAN_CASE_III_K1 if nobs <= size)
    values = _NARAYAN_CASE_III_K1[table_n]
    return {
        "source": "Narayan (2005), Case III, k=1",
        "table_n": table_n,
        "critical_10_i0": values[0],
        "critical_10_i1": values[1],
        "critical_05_i0": values[2],
        "critical_05_i1": values[3],
        "critical_01_i0": values[4],
        "critical_01_i1": values[5],
    }


def _bounds_decision(statistic: float, lower: float, upper: float) -> str:
    if statistic > upper:
        return "cointegrated"
    if statistic < lower:
        return "no_cointegration"
    return "inconclusive"


def _bootstrap_f_statistics(
    estimate: H1Estimate,
    replications: int,
    seed: int,
) -> np.ndarray:
    """Simulate the no-level-relationship null using independent random walks."""
    if replications < 1:
        return np.empty(0)
    base = estimate.prepared.dropna(subset=[A_LEVEL, K_LEVEL]).reset_index(drop=True)
    delta_a = np.diff(base[A_LEVEL].to_numpy(dtype=float))
    delta_k = np.diff(base[K_LEVEL].to_numpy(dtype=float))
    if len(delta_a) < 5 or len(delta_k) < 5:
        raise ValueError("Too few level observations for bootstrap bounds inference")
    delta_a -= delta_a.mean()
    delta_k -= delta_k.mean()
    rng = np.random.default_rng(seed)
    statistics: list[float] = []
    for _ in range(replications):
        simulated = base[["date_us", *estimate.spec.controls]].copy()
        a_increments = rng.choice(delta_a, size=len(base) - 1, replace=True)
        k_increments = rng.choice(delta_k, size=len(base) - 1, replace=True)
        simulated[A_LEVEL] = np.r_[base[A_LEVEL].iloc[0], base[A_LEVEL].iloc[0] + np.cumsum(a_increments)]
        simulated[K_LEVEL] = np.r_[base[K_LEVEL].iloc[0], base[K_LEVEL].iloc[0] + np.cumsum(k_increments)]
        prepared = prepare_uecm_data(
            simulated,
            max_p=estimate.spec.p_order,
            max_q=estimate.spec.q_order,
            controls=estimate.spec.controls,
        )
        try:
            fit, _ = _fit_order(prepared, estimate.spec, hac_lags=None)
            test = fit.f_test(f"{A_LAG} = 0, {K_LAG} = 0")
            statistic = float(np.asarray(test.fvalue).squeeze())
        except (ValueError, np.linalg.LinAlgError):
            continue
        if np.isfinite(statistic):
            statistics.append(statistic)
    return np.asarray(statistics, dtype=float)


def bounds_test(
    estimate: H1Estimate,
    *,
    bootstrap_replications: int = 499,
    seed: int = 20261123,
) -> dict[str, float | int | str | None]:
    """Run the level-joint F test, Narayan bounds, and bootstrap robustness test."""
    test = estimate.fit.f_test(f"{A_LAG} = 0, {K_LAG} = 0")
    f_statistic = float(np.asarray(test.fvalue).squeeze())
    output: dict[str, float | int | str | None] = {
        "f_statistic": f_statistic,
        "f_p_value_standard": float(np.asarray(test.pvalue).squeeze()),
        "nobs": int(estimate.fit.nobs),
        "case": "III: unrestricted intercept, no trend",
        "levels_regressors_k": 1,
        "narayan_applicable": False,
        "narayan_decision_05": None,
    }
    critical = narayan_case_iii_bounds(int(estimate.fit.nobs))
    if critical is not None:
        output.update(critical)
        output["narayan_applicable"] = True
        output["narayan_decision_05"] = _bounds_decision(
            f_statistic,
            float(critical["critical_05_i0"]),
            float(critical["critical_05_i1"]),
        )
        output["primary_bounds_source"] = critical["source"]
        output["primary_bounds_decision_05"] = output["narayan_decision_05"]
    else:
        values = _PSS_CASE_III_K1
        output.update({
            "primary_bounds_source": "Pesaran, Shin, and Smith (2001), Case III, k=1, asymptotic",
            "critical_10_i0": values[0],
            "critical_10_i1": values[1],
            "critical_05_i0": values[2],
            "critical_05_i1": values[3],
            "critical_01_i0": values[4],
            "critical_01_i1": values[5],
            "primary_bounds_decision_05": _bounds_decision(
                f_statistic, values[2], values[3]
            ),
        })

    bootstrap = _bootstrap_f_statistics(estimate, bootstrap_replications, seed)
    output["bootstrap_requested"] = bootstrap_replications
    output["bootstrap_successful"] = int(len(bootstrap))
    if len(bootstrap):
        output["bootstrap_p_value"] = float(
            (1 + np.count_nonzero(bootstrap >= f_statistic)) / (len(bootstrap) + 1)
        )
        output["bootstrap_critical_10"] = float(np.quantile(bootstrap, 0.90))
        output["bootstrap_critical_05"] = float(np.quantile(bootstrap, 0.95))
        output["bootstrap_critical_01"] = float(np.quantile(bootstrap, 0.99))
        output["bootstrap_decision_05"] = (
            "cointegrated"
            if f_statistic > float(output["bootstrap_critical_05"])
            else "do_not_reject_no_cointegration"
        )
    else:
        output.update({
            "bootstrap_p_value": None,
            "bootstrap_critical_10": None,
            "bootstrap_critical_05": None,
            "bootstrap_critical_01": None,
            "bootstrap_decision_05": None,
        })
    return output


def rolling_stability(
    frame: pd.DataFrame,
    controls: Iterable[str] = (),
    *,
    window: int = 60,
    step: int = 1,
    max_p: int = 3,
    max_q: int = 3,
) -> pd.DataFrame:
    """Re-estimate theta, the equilibrium intercept, and ECT persistence by window."""
    ordered = _levels(frame)
    if window < 20:
        raise ValueError("rolling H1 window must contain at least 20 observations")
    if step < 1:
        raise ValueError("rolling H1 step must be positive")
    rows: list[dict[str, object]] = []
    for end in range(window, len(ordered) + 1, step):
        one = ordered.iloc[end - window:end].copy()
        try:
            estimate = estimate_h1(
                one, controls, max_p=max_p, max_q=max_q, hac_lags=3
            )
        except ValueError:
            continue
        rows.append({
            "window_start": one["date_us"].iloc[0],
            "window_end": one["date_us"].iloc[-1],
            **estimate.summary,
        })
    return pd.DataFrame(rows)


def rolling_equilibrium_series(
    frame: pd.DataFrame,
    rolling: pd.DataFrame,
) -> pd.DataFrame:
    """Combine rolling H1 parameters with the observed levels and ECT."""
    if rolling.empty:
        raise ValueError("Cannot plot rolling H1 results because no windows were estimated")
    required = {"window_start", "window_end", "intercept", "theta"}
    if missing := required.difference(rolling.columns):
        raise ValueError(f"Rolling H1 results are missing columns: {sorted(missing)}")

    levels = _levels(frame)[["date_us", A_LEVEL, K_LEVEL]].copy()
    levels["date_us"] = pd.to_datetime(levels["date_us"])
    series = rolling.copy()
    series["date_us"] = pd.to_datetime(series["window_end"])
    series = series.merge(levels, on="date_us", how="left", validate="one_to_one")
    if series[[A_LEVEL, K_LEVEL]].isna().any().any():
        raise ValueError("Rolling H1 window endpoints do not match the daily level panel")
    series["equilibrium"] = (
        pd.to_numeric(series["intercept"], errors="coerce")
        + pd.to_numeric(series["theta"], errors="coerce") * series[K_LEVEL]
    )
    series["ect"] = series[A_LEVEL] - series["equilibrium"]
    return series.sort_values("date_us").reset_index(drop=True)


def plot_rolling_equilibrium(
    frame: pd.DataFrame,
    rolling: pd.DataFrame,
    output: Path,
) -> Path:
    """Plot the observed level, rolling equilibrium, and rolling ECT over time."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    series = rolling_equilibrium_series(frame, rolling)
    dates = pd.to_datetime(series["date_us"])
    actual = series[A_LEVEL].to_numpy(dtype=float)
    equilibrium = series["equilibrium"].to_numpy(dtype=float)
    ect = series["ect"].to_numpy(dtype=float)

    figure, (level_axis, ect_axis) = plt.subplots(
        2,
        1,
        figsize=(11, 7.5),
        sharex=True,
        gridspec_kw={"height_ratios": (1.35, 1.0), "hspace": 0.12},
    )
    level_axis.plot(dates, actual, color="#1f77b4", linewidth=2.0, label=r"Actual $a_d$")
    level_axis.plot(
        dates,
        equilibrium,
        color="#ff7f0e",
        linewidth=2.0,
        label=r"Rolling equilibrium $\hat{a}_d^*$",
    )
    level_axis.fill_between(
        dates, actual, equilibrium, color="#7f7f7f", alpha=0.12, label=r"$ECT_d$ gap"
    )
    level_axis.set_ylabel("Log USD price level")
    level_axis.set_title("SKHY level, rolling long-run equilibrium, and equilibrium error")
    level_axis.legend(frameon=False, ncol=3, loc="best")
    level_axis.grid(axis="y", alpha=0.25)

    ect_axis.axhline(0.0, color="#333333", linewidth=1.0, label="Equilibrium")
    ect_axis.plot(dates, ect, color="#2ca02c", linewidth=1.8, marker="o", markersize=3)
    ect_axis.fill_between(
        dates, 0.0, ect, where=ect >= 0.0, color="#d62728", alpha=0.18,
        interpolate=True, label="SKHY above equilibrium",
    )
    ect_axis.fill_between(
        dates, 0.0, ect, where=ect < 0.0, color="#1f77b4", alpha=0.18,
        interpolate=True, label="SKHY below equilibrium",
    )
    ect_axis.set_ylabel(r"Rolling $ECT_d$")
    ect_axis.set_xlabel("Daily date")
    ect_axis.grid(axis="y", alpha=0.25)
    ect_axis.legend(frameon=False, ncol=3, loc="best")
    ect_axis.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=9))
    ect_axis.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ect_axis.xaxis.get_major_locator()))

    figure.text(
        0.01,
        0.01,
        "Each point uses the ARDL parameters estimated in the rolling window ending on that date.",
        fontsize=9,
        color="#555555",
    )
    figure.align_ylabels((level_axis, ect_axis))
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=H2_FEATURE_PATH)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "features" / "experiments" / "run_id=h1-daily-ardl")
    parser.add_argument("--max-p", type=int, default=3)
    parser.add_argument("--max-q", type=int, default=3)
    parser.add_argument("--controls", nargs="*", default=())
    parser.add_argument("--rolling-window", type=int, default=60)
    parser.add_argument("--rolling-step", type=int, default=1)
    parser.add_argument("--bootstrap-replications", type=int, default=499)
    parser.add_argument("--seed", type=int, default=20261123)
    parser.add_argument(
        "--plot-output",
        type=Path,
        default=PROJECT_ROOT / "reports" / "h1_rolling_equilibrium.png",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    frame = pd.read_parquet(args.input)
    estimate = estimate_h1(
        frame, args.controls, max_p=args.max_p, max_q=args.max_q
    )
    inference = bounds_test(
        estimate,
        bootstrap_replications=args.bootstrap_replications,
        seed=args.seed,
    )
    rolling = rolling_stability(
        frame,
        args.controls,
        window=args.rolling_window,
        step=args.rolling_step,
        max_p=args.max_p,
        max_q=args.max_q,
    )
    output_dir = feature_output_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([estimate.summary]).to_parquet(output_dir / "h1_model.parquet", index=False)
    pd.DataFrame([inference]).to_parquet(output_dir / "h1_bounds.parquet", index=False)
    rolling.to_parquet(output_dir / "h1_rolling_stability.parquet", index=False)
    plot_path = plot_rolling_equilibrium(frame, rolling, args.plot_output)
    ect = _levels(frame)[["date_us", A_LEVEL, K_LEVEL]].copy()
    ect["ect"] = equilibrium_error(frame, estimate.summary)
    ect.to_parquet(output_dir / "h1_ect.parquet", index=False)
    print(pd.Series({**estimate.summary, **inference}).to_string(), flush=True)
    print(f"[H1] Wrote ARDL artifacts to {output_dir}", flush=True)
    print(f"[H1] Wrote rolling-equilibrium plot to {plot_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
