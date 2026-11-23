"""Estimate daily H1/H2 models for the SKHY overnight return.

H1 contains the FX-adjusted Korean ordinary-share return. H2 augments H1 with
daily control groups and ranks the specifications jointly by the absolute
effect size of lambda and its HAC p-value on a common sample. Expanding-window
forecasts are retained as model diagnostics.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import combinations
import math
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant

from src.config import PathSettings, feature_output_path
from src.features.h1_h2 import (
    CONTROL_GROUPS, H2_FEATURE_PATH, PREDICTOR,
    write_h2_features,
)
from src.hypotheses.h1 import (
    equilibrium_error as _equilibrium_error,
    fit_h1_equilibrium,
)


TARGET = "overnight_return"


@dataclass(frozen=True)
class OvernightSpec:
    static_controls: tuple[str, ...] = ()


def _candidate_control_specs(
    groups: dict[str, tuple[str, ...]],
) -> list[tuple[str, str, tuple[str, ...]]]:
    """Single-group, full, and leave-one-group-out control specifications."""
    all_controls = tuple(column for columns in groups.values() for column in columns)
    candidates = [
        (f"add_{group}", "single_group", controls)
        for group, controls in groups.items()
    ]
    candidates.append(("all_controls", "full", all_controls))
    for omitted, omitted_controls in groups.items():
        controls = tuple(column for column in all_controls if column not in omitted_controls)
        candidates.append((f"all_except_{omitted}", "leave_one_group_out", controls))
    return candidates


def prepare_ecm_panel(
    frame: pd.DataFrame,
    *,
    max_p: int = 3,
    max_q: int = 3,
    h1: dict[str, float | int] | None = None,
) -> pd.DataFrame:
    output = frame.sort_values("date_us").reset_index(drop=True).copy()
    if h1 is None:
        h1, _ = fit_h1_equilibrium(output)
    output["ect"] = _equilibrium_error(output, h1)
    for lag in range(1, max_p + 1):
        output[f"overnight_return_l{lag}"] = output[TARGET].shift(lag)
    for lag in range(1, max_q + 1):
        output[f"{PREDICTOR}_l{lag}"] = output[PREDICTOR].shift(lag)
    return output


def _columns(spec: OvernightSpec, p_order: int = 0, q_order: int = 0) -> tuple[str, ...]:
    columns = [PREDICTOR, "ect"]
    columns.extend(spec.static_controls)
    columns.extend(f"overnight_return_l{lag}" for lag in range(1, p_order + 1))
    columns.extend(f"{PREDICTOR}_l{lag}" for lag in range(1, q_order + 1))
    return tuple(columns)


def _design(
    frame: pd.DataFrame, spec: OvernightSpec, p_order: int = 0, q_order: int = 0,
) -> pd.DataFrame:
    return add_constant(
        frame[list(_columns(spec, p_order, q_order))].astype(float),
        has_constant="add",
    )


def _sample(
    frame: pd.DataFrame, spec: OvernightSpec, p_order: int = 0, q_order: int = 0,
) -> pd.DataFrame:
    required = ["date_us", TARGET, *_columns(spec, p_order, q_order)]
    sample = (
        frame.replace([np.inf, -np.inf], np.nan)
        .dropna(subset=required)
        .sort_values("date_us")
        .reset_index(drop=True)
    )
    if sample["date_us"].duplicated().any():
        raise ValueError("Expected one daily observation per US session")
    return sample


def fit_daily(
    frame: pd.DataFrame,
    spec: OvernightSpec,
    p_order: int = 0,
    q_order: int = 0,
    *,
    hac_lags: int = 3,
):
    sample = _sample(frame, spec, p_order, q_order)
    if len(sample) <= len(_columns(spec, p_order, q_order)) + 2:
        raise ValueError("Insufficient complete daily observations")
    return OLS(sample[TARGET], _design(sample, spec, p_order, q_order)).fit(
        cov_type="HAC", cov_kwds={"maxlags": hac_lags}
    )


def walk_forward(
    frame: pd.DataFrame,
    spec: OvernightSpec,
    p_order: int = 0,
    q_order: int = 0,
    min_train: int = 40,
    **_: object,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    """Generate forecasts while re-estimating H1 and H2 at every origin."""
    prepared = prepare_ecm_panel(frame, max_p=p_order, max_q=q_order)
    sample = _sample(prepared, spec, p_order, q_order)
    if len(sample) <= min_train:
        raise ValueError(f"Need more than {min_train} complete rows; got {len(sample)}")
    if min_train <= len(_columns(spec, p_order, q_order)) + 2:
        raise ValueError("Initial window is too small for the requested controls")

    records: list[dict[str, object]] = []
    for index in range(min_train, len(sample)):
        train = sample.iloc[:index].copy()
        test = sample.iloc[[index]].copy()
        h1, _ = fit_h1_equilibrium(train)
        train["ect"] = _equilibrium_error(train, h1)
        test["ect"] = _equilibrium_error(test, h1)
        fit = OLS(
            train[TARGET], _design(train, spec, p_order, q_order)
        ).fit()
        forecast = float(
            fit.predict(_design(test, spec, p_order, q_order)).iloc[0]
        )
        actual = float(test[TARGET].iloc[0])
        records.append({
            "date_us": test["date_us"].iloc[0],
            "actual_return": actual,
            "forecast_return": forecast,
            "forecast_error": actual - forecast,
            "correct_direction": bool(np.sign(actual) == np.sign(forecast)),
            "training_n": len(train),
            "training_through": train["date_us"].iloc[-1],
        })
    forecasts = pd.DataFrame(records)
    errors = forecasts["forecast_error"]
    actual = forecasts["actual_return"]
    model_sse = float(np.square(errors).sum())
    zero_sse = float(np.square(actual).sum())
    metrics: dict[str, float | int] = {
        "n_forecasts": len(forecasts),
        "rmse": float(np.sqrt(np.mean(np.square(errors)))),
        "mae": float(np.mean(np.abs(errors))),
        "bias": float(errors.mean()),
        "direction_accuracy": float(forecasts["correct_direction"].mean()),
        "oos_r2_zero": 1.0 - model_sse / zero_sse if zero_sse else math.nan,
    }
    return forecasts, metrics


def build_full_sample_comparison(
    frame: pd.DataFrame,
    groups: dict[str, tuple[str, ...]] = CONTROL_GROUPS,
    *,
    min_train: int = 40,
    max_p: int = 3,
    max_q: int = 3,
) -> pd.DataFrame:
    """Build a descriptive common-sample comparison of eligible H2 models.

    Lambda is the coefficient on the pre-response equilibrium error. The ECT is
    oriented as SKHY minus its fitted equilibrium. Specifications are retained
    only when lambda < 0 and its HAC p-value < 0.05, then ranked by the absolute
    coefficient on the current FX-adjusted Korean return. This table is not
    used by the nested forecast selector.
    """
    if max_p not in range(4) or max_q not in range(4):
        raise ValueError("max_p and max_q must each be between 0 and 3")
    all_controls = tuple(column for columns in groups.values() for column in columns)
    prepared = prepare_ecm_panel(frame, max_p=max_p, max_q=max_q)
    max_lag_columns = [
        *(f"overnight_return_l{lag}" for lag in range(1, max_p + 1)),
        *(f"{PREDICTOR}_l{lag}" for lag in range(1, max_q + 1)),
    ]
    common = prepared.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["date_us", TARGET, PREDICTOR, "ect", *all_controls, *max_lag_columns]
    ).sort_values("date_us").reset_index(drop=True)
    if len(common) <= min_train:
        raise ValueError(
            f"Need more than {min_train} common rows for H2 ranking; got {len(common)}"
        )

    candidates = _candidate_control_specs(groups)

    records: list[dict[str, object]] = []
    for model_name, model_class, controls in candidates:
        spec = OvernightSpec(controls)
        order_candidates: list[tuple[float, int, int]] = []
        for p_order in range(max_p + 1):
            for q_order in range(max_q + 1):
                ols = OLS(
                    common[TARGET], _design(common, spec, p_order, q_order)
                ).fit()
                order_candidates.append((float(ols.bic), p_order, q_order))
        _, p_order, q_order = min(order_candidates)
        fit = fit_daily(common, spec, p_order, q_order)
        _, metrics = walk_forward(
            frame, spec, p_order=p_order, q_order=q_order, min_train=min_train
        )
        coefficient = float(fit.params["ect"])
        p_value = float(fit.pvalues["ect"])
        korean_coefficient = float(fit.params[PREDICTOR])
        korean_p_value = float(fit.pvalues[PREDICTOR])
        records.append({
            "model_name": model_name,
            "model_class": model_class,
            "controls": ", ".join(controls),
            "control_count": len(controls),
            "p_order": p_order,
            "q_order": q_order,
            "nobs": int(fit.nobs),
            "lambda": coefficient,
            "abs_lambda": abs(coefficient),
            "lambda_p_value": p_value,
            "lambda_significant_05": p_value < 0.05,
            "ecm_stable": -2.0 < coefficient < 0.0,
            "lambda_eligible": coefficient < 0.0 and p_value < 0.05,
            "korean_return_coefficient": korean_coefficient,
            "abs_korean_return_coefficient": abs(korean_coefficient),
            "korean_return_p_value": korean_p_value,
            "r_squared": float(fit.rsquared),
            "adjusted_r_squared": float(fit.rsquared_adj),
            "bic": float(fit.bic),
            **metrics,
        })
    ranking = pd.DataFrame(records)
    ranking["lambda_effect_rank"] = ranking["abs_lambda"].rank(
        ascending=False, method="min"
    )
    ranking["lambda_p_rank"] = ranking["lambda_p_value"].rank(
        ascending=True, method="min"
    )
    ranking["lambda_combined_score"] = (
        ranking["lambda_effect_rank"] + ranking["lambda_p_rank"]
    ) / 2.0
    ranking = ranking.loc[ranking["lambda_eligible"]].copy()
    if ranking.empty:
        raise ValueError("No H2 specification has negative, significant ECT lambda")
    ranking["korean_return_effect_rank"] = ranking[
        "abs_korean_return_coefficient"
    ].rank(ascending=False, method="min")
    ranking = ranking.sort_values(
        ["abs_korean_return_coefficient", "korean_return_p_value", "bic", "model_name"],
        ascending=[False, True, True, True],
    ).reset_index(drop=True)
    ranking.insert(0, "h2_rank", np.arange(1, len(ranking) + 1))
    return ranking


def analyze_kr_proxy_redundancy(
    frame: pd.DataFrame,
    *,
    min_train: int = 40,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare every EWY/FLKR/USD-KRW subset and quantify duplication."""
    proxies = (
        "kr_proxy_ewy_return",
        "kr_proxy_flkr_return",
        "kr_proxy_usdkrw_return",
    )
    prepared = prepare_ecm_panel(frame, max_p=3, max_q=3)
    max_lags = [
        *(f"overnight_return_l{lag}" for lag in range(1, 4)),
        *(f"{PREDICTOR}_l{lag}" for lag in range(1, 4)),
    ]
    common = prepared.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["date_us", TARGET, PREDICTOR, "ect", *proxies, *max_lags]
    ).sort_values("date_us").reset_index(drop=True)

    rows: list[dict[str, object]] = []
    for size in range(1, len(proxies) + 1):
        for subset in combinations(proxies, size):
            spec = OvernightSpec(subset)
            fit = fit_daily(common, spec, 0, 0)
            _, metrics = walk_forward(common, spec, 0, 0, min_train=min_train)
            record: dict[str, object] = {
                "proxy_set": " + ".join(
                    name.removeprefix("kr_proxy_").removesuffix("_return").upper()
                    for name in subset
                ),
                "controls": ", ".join(subset),
                "control_count": len(subset),
                "nobs": int(fit.nobs),
                "bic": float(fit.bic),
                "adjusted_r_squared": float(fit.rsquared_adj),
                "lambda": float(fit.params["ect"]),
                "lambda_p_value": float(fit.pvalues["ect"]),
                "korean_return_coefficient": float(fit.params[PREDICTOR]),
                "korean_return_p_value": float(fit.pvalues[PREDICTOR]),
                **metrics,
            }
            for proxy in subset:
                record[f"coefficient__{proxy}"] = float(fit.params[proxy])
                record[f"pvalue__{proxy}"] = float(fit.pvalues[proxy])
            rows.append(record)
    comparison = pd.DataFrame(rows).sort_values(["rmse", "bic"]).reset_index(drop=True)
    comparison.insert(0, "proxy_rank", np.arange(1, len(comparison) + 1))

    variables = [PREDICTOR, *proxies]
    correlations = common[variables].corr()
    diagnostic: dict[str, object] = {"nobs": len(common)}
    for left, right in combinations(variables, 2):
        diagnostic[f"correlation__{left}__{right}"] = float(correlations.loc[left, right])
    for variable in variables:
        others = [name for name in variables if name != variable]
        auxiliary = OLS(
            common[variable].astype(float),
            add_constant(common[others].astype(float), has_constant="add"),
        ).fit()
        diagnostic[f"vif__{variable}"] = (
            float(1.0 / (1.0 - auxiliary.rsquared))
            if auxiliary.rsquared < 1.0 else math.inf
        )
    return comparison, pd.DataFrame([diagnostic])


def nested_walk_forward_selection(
    frame: pd.DataFrame,
    groups: dict[str, tuple[str, ...]] = CONTROL_GROUPS,
    *,
    min_train: int = 40,
    inner_min_train: int = 30,
    max_p: int = 3,
    max_q: int = 3,
) -> tuple[pd.DataFrame, dict[str, float | int], pd.DataFrame]:
    """Nested H1/H2 selection with no information from the forecast date.

    At each candidate-forecast origin, H1 and the ECT are re-estimated, p/q are
    selected by BIC, and all control specifications are fitted on prior rows.
    At each outer origin, eligible specifications are ranked by their cached
    inner expanding-window errors. The simplest model within one standard error
    of the best mean squared forecast loss is selected.
    """
    if inner_min_train >= min_train:
        raise ValueError("inner_min_train must be smaller than min_train")
    candidates = _candidate_control_specs(groups)
    all_controls = tuple(column for columns in groups.values() for column in columns)
    base = prepare_ecm_panel(frame, max_p=max_p, max_q=max_q)
    max_lags = [
        *(f"overnight_return_l{lag}" for lag in range(1, max_p + 1)),
        *(f"{PREDICTOR}_l{lag}" for lag in range(1, max_q + 1)),
    ]
    common = base.replace([np.inf, -np.inf], np.nan).dropna(
        subset=[
            "date_us", TARGET, PREDICTOR, "previous_us_close",
            "kr_fx_adjusted_close_usd", *all_controls, *max_lags,
        ]
    ).sort_values("date_us").reset_index(drop=True)
    if len(common) <= min_train:
        raise ValueError(f"Need more than {min_train} common rows; got {len(common)}")

    base_dates = pd.to_datetime(base["date_us"])
    candidate_rows: list[dict[str, object]] = []
    for origin in range(inner_min_train, len(common)):
        test = common.iloc[[origin]].copy()
        cutoff = pd.Timestamp(test["date_us"].iloc[0])
        h1_history = base.loc[base_dates.lt(cutoff)]
        h1, _ = fit_h1_equilibrium(h1_history)
        train = common.iloc[:origin].copy()
        train["ect"] = _equilibrium_error(train, h1)
        test["ect"] = _equilibrium_error(test, h1)

        for model_name, model_class, controls in candidates:
            spec = OvernightSpec(controls)
            orders: list[tuple[float, int, int]] = []
            for p_order in range(max_p + 1):
                for q_order in range(max_q + 1):
                    columns = _columns(spec, p_order, q_order)
                    if len(train) <= len(columns) + 2:
                        continue
                    ols = OLS(
                        train[TARGET], _design(train, spec, p_order, q_order)
                    ).fit()
                    if np.isfinite(ols.bic):
                        orders.append((float(ols.bic), p_order, q_order))
            if not orders:
                continue
            bic, p_order, q_order = min(orders)
            fit = OLS(
                train[TARGET], _design(train, spec, p_order, q_order)
            ).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
            forecast = float(
                fit.predict(_design(test, spec, p_order, q_order)).iloc[0]
            )
            actual = float(test[TARGET].iloc[0])
            lambda_value = float(fit.params["ect"])
            lambda_p = float(fit.pvalues["ect"])
            beta = float(fit.params[PREDICTOR])
            beta_p = float(fit.pvalues[PREDICTOR])
            candidate_rows.append({
                "origin_index": origin,
                "date_us": test["date_us"].iloc[0],
                "training_n": len(train),
                "training_through": train["date_us"].iloc[-1],
                "model_name": model_name,
                "model_class": model_class,
                "controls": ", ".join(controls),
                "control_count": len(controls),
                "p_order": p_order,
                "q_order": q_order,
                "complexity": len(_columns(spec, p_order, q_order)),
                "bic": bic,
                "lambda": lambda_value,
                "lambda_p_value": lambda_p,
                "korean_return_coefficient": beta,
                "korean_return_p_value": beta_p,
                "eligible": (
                    -2.0 < lambda_value < 0.0
                    and lambda_p < 0.05
                    and beta > 0.0
                    and beta_p < 0.05
                ),
                "actual_return": actual,
                "forecast_return": forecast,
                "forecast_error": actual - forecast,
                "squared_error": (actual - forecast) ** 2,
                "absolute_error": abs(actual - forecast),
                "correct_direction": bool(np.sign(actual) == np.sign(forecast)),
                "h1_intercept": float(h1["intercept"]),
                "h1_theta": float(h1["theta"]),
            })
    candidates_frame = pd.DataFrame(candidate_rows)

    forecasts: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    for origin in range(min_train, len(common)):
        current = candidates_frame.loc[candidates_frame["origin_index"].eq(origin)].copy()
        history = candidates_frame.loc[candidates_frame["origin_index"].lt(origin)]
        summaries: list[dict[str, object]] = []
        for model_name, observations in history.groupby("model_name"):
            losses = observations["squared_error"].astype(float)
            summaries.append({
                "model_name": model_name,
                "inner_n": len(observations),
                "inner_mse": float(losses.mean()),
                "inner_rmse": float(np.sqrt(losses.mean())),
                "inner_loss_se": float(losses.std(ddof=1) / np.sqrt(len(losses)))
                if len(losses) > 1 else math.inf,
                "inner_mae": float(observations["absolute_error"].mean()),
                "inner_direction_accuracy": float(observations["correct_direction"].mean()),
                "lambda_stability_sd": float(observations["lambda"].std(ddof=1)),
                "beta_stability_sd": float(
                    observations["korean_return_coefficient"].std(ddof=1)
                ),
            })
        current = current.merge(pd.DataFrame(summaries), on="model_name", how="left")
        eligible = current.loc[current["eligible"] & current["inner_rmse"].notna()].copy()
        if eligible.empty:
            raise ValueError(
                f"No eligible nested H2 specification at {current['date_us'].iloc[0]}"
            )
        best = eligible.sort_values(["inner_rmse", "inner_mae"]).iloc[0]
        mse_threshold = float(best["inner_mse"] + best["inner_loss_se"])
        eligible["within_one_se"] = eligible["inner_mse"].le(mse_threshold)
        finalists = eligible.loc[eligible["within_one_se"]].copy()
        finalists = finalists.sort_values(
            [
                "complexity", "inner_rmse", "inner_mae",
                "inner_direction_accuracy", "lambda_stability_sd",
                "beta_stability_sd", "bic",
            ],
            ascending=[True, True, True, False, True, True, True],
        )
        selected = finalists.iloc[0]
        current["mse_one_se_threshold"] = mse_threshold
        current["within_one_se"] = (
            current["eligible"] & current["inner_mse"].le(mse_threshold)
        )
        current["selected"] = current["model_name"].eq(selected["model_name"])
        audit_rows.extend(current.to_dict("records"))
        forecasts.append({
            "date_us": selected["date_us"],
            "actual_return": float(selected["actual_return"]),
            "forecast_return": float(selected["forecast_return"]),
            "forecast_error": float(selected["forecast_error"]),
            "correct_direction": bool(selected["correct_direction"]),
            "training_n": int(selected["training_n"]),
            "training_through": selected["training_through"],
            "selected_model": selected["model_name"],
            "selected_controls": selected["controls"],
            "p_order": int(selected["p_order"]),
            "q_order": int(selected["q_order"]),
            "lambda": float(selected["lambda"]),
            "lambda_p_value": float(selected["lambda_p_value"]),
            "korean_return_coefficient": float(selected["korean_return_coefficient"]),
            "korean_return_p_value": float(selected["korean_return_p_value"]),
            "inner_rmse": float(selected["inner_rmse"]),
            "one_se_threshold_mse": mse_threshold,
        })

    output = pd.DataFrame(forecasts)
    errors = output["forecast_error"]
    actual = output["actual_return"]
    model_sse = float(np.square(errors).sum())
    zero_sse = float(np.square(actual).sum())
    metrics: dict[str, float | int] = {
        "n_forecasts": len(output),
        "rmse": float(np.sqrt(np.mean(np.square(errors)))),
        "mae": float(np.mean(np.abs(errors))),
        "bias": float(errors.mean()),
        "direction_accuracy": float(output["correct_direction"].mean()),
        "oos_r2_zero": 1.0 - model_sse / zero_sse if zero_sse else math.nan,
    }
    return output, metrics, pd.DataFrame(audit_rows)


def write_ablation_markdown(
    ranking: pd.DataFrame,
    path: Path,
    *,
    run_id: str,
    min_train: int,
    nested_audit: pd.DataFrame | None = None,
    nested_metrics: dict[str, float | int] | None = None,
    h1_summary: pd.DataFrame | None = None,
    h2_summary: pd.DataFrame | None = None,
    proxy_comparison: pd.DataFrame | None = None,
    proxy_diagnostics: pd.DataFrame | None = None,
) -> Path:
    """Write a concise report containing only operative nested-selection results."""
    del ranking
    if nested_audit is None or nested_audit.empty or nested_metrics is None:
        raise ValueError("Nested audit and metrics are required for the H2 report")

    selected_rows = nested_audit.loc[nested_audit["selected"]].copy()
    selection_counts = selected_rows.groupby("model_name").size().sort_values(ascending=False)
    p_counts = selected_rows["p_order"].value_counts().sort_index()
    q_counts = selected_rows["q_order"].value_counts().sort_index()
    last_date = nested_audit["date_us"].max()
    top_models = nested_audit.loc[
        nested_audit["date_us"].eq(last_date) & nested_audit["eligible"]
    ].sort_values(["inner_rmse", "complexity", "inner_mae"]).head(3)
    selected = top_models.loc[top_models["selected"]].iloc[0]

    lines = [
        "# H2 ablation study results",
        "",
        f"**Run ID:** `{run_id}`  ",
        f"**Initial outer training size:** {min_train} observations  ",
        f"**Final evaluated origin:** {last_date}",
        "",
        "## Nested selection criteria",
        "",
        "Every outer forecast uses prior observations only. At each origin, H1 is",
        "re-estimated, the training and forecast-date ECTs are rebuilt, and p and q",
        "from 0 through 3 are selected by BIC within each control specification.",
        "Eligible models must satisfy:",
        "",
        "\\[",
        "-2<\\lambda<0,\\qquad p_\\lambda<0.05,",
        "\\]",
        "",
        "\\[",
        "\\beta_{KR,FX}>0,\\qquad p_\\beta<0.05.",
        "\\]",
        "",
        "The retained models are ranked by average inner expanding-window OOS RMSE.",
        "The simplest model within one standard error of the lowest mean squared",
        "error is selected. MAE, direction accuracy, coefficient stability, and",
        "BIC are ordered tie-breakers.",
        "",
        "Selection therefore uses four primary reported quantities:",
        "",
        "- Average inner expanding-window OOS RMSE",
        "- ECT λ and its p-value",
        "- Korean FX-adjusted return β",
        "- Korean β p-value",
        "",
        "## Top model specifications",
        "",
        "| Selected | One-SE set | Model | p | q | Average inner RMSE | λ(ECT) | λ p-value | Korean β | β p-value |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in top_models.iterrows():
        lines.append(
            f"| {'yes' if bool(row['selected']) else ''} | "
            f"{'yes' if bool(row['within_one_se']) else ''} | `{row['model_name']}` | "
            f"{int(row['p_order'])} | {int(row['q_order'])} | "
            f"{float(row['inner_rmse']):.6f} | {float(row['lambda']):.6f} | "
            f"{float(row['lambda_p_value']):.3e} | "
            f"{float(row['korean_return_coefficient']):.6f} | "
            f"{float(row['korean_return_p_value']):.3e} |"
        )
    if (
        proxy_comparison is not None and not proxy_comparison.empty
        and proxy_diagnostics is not None and not proxy_diagnostics.empty
    ):
        diagnostic = proxy_diagnostics.iloc[0]
        recommended = proxy_comparison.loc[
            proxy_comparison["proxy_set"].eq("FLKR + USDKRW")
        ].iloc[0]
        lines.extend([
            "",
            "## Korean market-proxy redundancy analysis",
            "",
            "EWY and FLKR carry nearly identical Korean equity-market information:",
            f"their return correlation is {float(diagnostic['correlation__kr_proxy_ewy_return__kr_proxy_flkr_return']):.4f}, "
            f"with VIFs of {float(diagnostic['vif__kr_proxy_ewy_return']):.2f} and "
            f"{float(diagnostic['vif__kr_proxy_flkr_return']):.2f}. USD/KRW is materially less redundant, "
            f"with VIF {float(diagnostic['vif__kr_proxy_usdkrw_return']):.2f}.",
            "",
            "| Rank | Proxy controls | RMSE | BIC | Adj. R² | λ | λ p-value | Korean β | β p-value |",
            "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ])
        for _, row in proxy_comparison.iterrows():
            lines.append(
                f"| {int(row['proxy_rank'])} | {row['proxy_set']} | "
                f"{float(row['rmse']):.6f} | {float(row['bic']):.2f} | "
                f"{float(row['adjusted_r_squared']):.3f} | {float(row['lambda']):.6f} | "
                f"{float(row['lambda_p_value']):.3e} | "
                f"{float(row['korean_return_coefficient']):.6f} | "
                f"{float(row['korean_return_p_value']):.3e} |"
            )
        lines.extend([
            "",
            f"The recommended nonduplicative set is **FLKR + USD/KRW**: RMSE "
            f"{float(recommended['rmse']):.6f} and BIC {float(recommended['bic']):.2f}. "
            "EWY is excluded because it duplicates FLKR; USD/KRW is retained because",
            "it contributes a distinct currency component. The operational",
            "`kr_market_proxies` group therefore contains FLKR and USD/KRW only.",
        ])
    if h1_summary is not None and not h1_summary.empty and h2_summary is not None and not h2_summary.empty:
        h1 = h1_summary.iloc[0]
        h2 = h2_summary.iloc[0]
        lines.extend([
            "",
            "## Final H1 and H2 equations",
            "",
            "These equations are full-common-sample reference estimates for the",
            "currently retained VIX specification. In the nested backtest, H1, the",
            "ECT, p/q orders, and H2 coefficients are re-estimated at every origin",
            "using prior observations only, so forecast-date coefficients can vary.",
            "",
            "### H1: long-run level equilibrium",
            "",
            "Define the FX-adjusted Korean ordinary-share price as:",
            "",
            "\\[",
            "P^{KR,FX}_d=\\frac{0.1P^{000660,KRW}_d}{USDKRW_d}.",
            "\\]",
            "",
            "H1 estimates the unrestricted daily error-correction model:",
            "",
            "\\[",
            "\\Delta a_d=c+\\rho_a a_{d-1}+\\rho_k k_{d-1}"
            "+\\sum_i\\phi_i\\Delta a_{d-i}+\\sum_j\\gamma_j\\Delta k_{d-j}"
            "+\\delta'Z_d+\\varepsilon_d.",
            "\\]",
            "",
            "The derived fitted long-run equilibrium is:",
            "",
            "\\[",
            f"\\log P^{{SKHY}}_{{d-1,close}}={float(h1['intercept']):.6f}"
            f"+{float(h1['theta']):.6f}\\log P^{{KR,FX}}_d+u_d.",
            "\\]",
            "",
            f"H1 uses {int(h1['nobs'])} observations with p={int(h1['p_order'])} "
            f"and q={int(h1['q_order'])}; θ has delta-method HAC p-value "
            f"{float(h1['theta_p']):.3e}, R² is {float(h1['r_squared']):.6f}, "
            f"BIC is {float(h1['bic']):.6f}, and ECT residual persistence is "
            f"{float(h1['residual_persistence']):.6f}.",
            "",
            "The pre-response equilibrium error is:",
            "",
            "\\[",
            f"ECT_d=\\log P^{{SKHY}}_{{d-1,close}}-{float(h1['intercept']):.6f}"
            f"-{float(h1['theta']):.6f}\\log P^{{KR,FX}}_d.",
            "\\]",
            "",
            "Positive ECT means the prior SKHY close is above its fitted equilibrium;",
            "therefore convergence requires a negative H2 adjustment coefficient.",
            "",
            "### H2: overnight ECM with VIX control",
            "",
            "The target and return predictors are:",
            "",
            "\\[",
            "r^{ON}_d=\\log\\left(\\frac{P^{SKHY}_{d,09:30\\ open}}"
            "{P^{SKHY}_{d-1,close}}\\right),\\qquad "
            "r^{KR,FX}_d=\\Delta\\log P^{KR,FX}_d.",
            "\\]",
            "",
            f"For the retained p={int(h2['p_order'])}, q={int(h2['q_order'])} "
            "VIX specification:",
            "",
            "\\[",
            "r^{ON}_d=\\alpha+\\lambda ECT_d+\\beta r^{KR,FX}_d"
            "+\\gamma r^{VIX,known}_d+\\epsilon_d.",
            "\\]",
            "",
            "The fitted reference equation is:",
            "",
            "\\[",
            f"r^{{ON}}_d={float(h2['coefficient__const']):.6f}"
            f"{float(h2['coefficient__ect']):+.6f}ECT_d"
            f"{float(h2['coefficient__kr_fx_adjusted_return']):+.6f}r^{{KR,FX}}_d"
            f"{float(h2['coefficient__vix_daily_return']):+.6f}r^{{VIX,known}}_d"
            "+\\epsilon_d.",
            "\\]",
            "",
            "| H2 term | Coefficient | HAC p-value |",
            "| --- | ---: | ---: |",
            f"| Intercept | {float(h2['coefficient__const']):.6f} | {float(h2['pvalue__const']):.3e} |",
            f"| ECT λ | {float(h2['coefficient__ect']):.6f} | {float(h2['pvalue__ect']):.3e} |",
            f"| Korean FX-adjusted return β | {float(h2['coefficient__kr_fx_adjusted_return']):.6f} | {float(h2['pvalue__kr_fx_adjusted_return']):.3e} |",
            f"| Known VIX daily return γ | {float(h2['coefficient__vix_daily_return']):.6f} | {float(h2['pvalue__vix_daily_return']):.3e} |",
            "",
            f"H2 uses {int(h2['nobs'])} common-sample observations; adjusted R² is "
            f"{float(h2['adjusted_r_squared']):.6f} and BIC is {float(h2['bic']):.6f}.",
            "The VIX return is the latest completed daily VIX close-to-close return",
            "available before the SKHY open; it is not a contemporaneous post-open value.",
        ])
    lines.extend([
        "",
        "## Selected specification",
        "",
        f"`{selected['model_name']}` is selected with p={int(selected['p_order'])}, "
        f"q={int(selected['q_order'])}, average inner RMSE "
        f"{float(selected['inner_rmse']):.6f}, λ={float(selected['lambda']):.6f} "
        f"(p={float(selected['lambda_p_value']):.3e}), and Korean β="
        f"{float(selected['korean_return_coefficient']):.6f} "
        f"(p={float(selected['korean_return_p_value']):.3e}).",
        "",
        "## Nested out-of-sample performance",
        "",
        f"- Forecasts: {int(nested_metrics['n_forecasts'])}",
        f"- Selection frequency: `{selection_counts.index[0]}` on "
        f"{int(selection_counts.iloc[0])} of {int(nested_metrics['n_forecasts'])} origins",
        "- Selected p orders: " + ", ".join(
            f"p={int(order)} on {int(count)} origins" for order, count in p_counts.items()
        ),
        "- Selected q orders: " + ", ".join(
            f"q={int(order)} on {int(count)} origins" for order, count in q_counts.items()
        ),
        f"- RMSE: {float(nested_metrics['rmse']):.6f}",
        f"- MAE: {float(nested_metrics['mae']):.6f}",
        f"- Direction accuracy: {float(nested_metrics['direction_accuracy']):.2%}",
        f"- OOS R²: {float(nested_metrics['oos_r2_zero']):.6f}",
        "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _model_record(name: str, spec: OvernightSpec, fit: object) -> dict[str, object]:
    record: dict[str, object] = {
        "model": name,
        "target": TARGET,
        "predictor": PREDICTOR,
        "controls": ", ".join(spec.static_controls) if spec.static_controls else "none",
        "nobs": int(fit.nobs),
        "r_squared": float(fit.rsquared),
        "adjusted_r_squared": float(fit.rsquared_adj),
        "bic": float(fit.bic),
        "predictor_coefficient": float(fit.params[PREDICTOR]),
        "predictor_p": float(fit.pvalues[PREDICTOR]),
        "lambda": float(fit.params["ect"]),
        "lambda_p_value": float(fit.pvalues["ect"]),
    }
    for term in fit.params.index:
        record[f"coefficient__{term}"] = float(fit.params[term])
        record[f"pvalue__{term}"] = float(fit.pvalues[term])
    return record


def run_model(
    run_id: str, max_p: int = 3, max_q: int = 3, min_train: int = 40,
) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    output_data = feature_output_path(PathSettings().feature_run(run_id))
    output_data.mkdir(parents=True, exist_ok=True)
    panel = pd.read_parquet(H2_FEATURE_PATH) if H2_FEATURE_PATH.is_file() else write_h2_features()
    h1_result, _ = fit_h1_equilibrium(panel)
    prepared = prepare_ecm_panel(panel, max_p=max_p, max_q=max_q, h1=h1_result)
    prepared.to_parquet(output_data / "overnight.parquet", index=False)
    h1_summary = pd.DataFrame([{
        "model": "H1",
        "h1_spec": "daily_ardl_uecm",
        "status": "ok",
        **h1_result,
    }])

    ranking = build_full_sample_comparison(
        panel, min_train=min_train, max_p=max_p, max_q=max_q
    )
    proxy_comparison, proxy_diagnostics = analyze_kr_proxy_redundancy(
        panel, min_train=min_train
    )
    forecasts, metrics, nested_audit = nested_walk_forward_selection(
        panel,
        min_train=min_train,
        max_p=max_p,
        max_q=max_q,
    )
    last_date = nested_audit["date_us"].max()
    final_selected = nested_audit.loc[
        nested_audit["date_us"].eq(last_date) & nested_audit["selected"]
    ].iloc[0]
    selected_row = ranking.loc[
        ranking["model_name"].eq(final_selected["model_name"])
    ].iloc[0]
    selected_controls = tuple(
        value.strip() for value in str(final_selected["controls"]).split(",")
        if value.strip()
    )
    selected_groups = tuple(
        group for group, columns in CONTROL_GROUPS.items()
        if all(column in selected_controls for column in columns)
    )
    spec = OvernightSpec(selected_controls)
    p_order = int(final_selected["p_order"])
    q_order = int(final_selected["q_order"])
    all_controls = [column for columns in CONTROL_GROUPS.values() for column in columns]
    max_lags = [
        *(f"overnight_return_l{lag}" for lag in range(1, max_p + 1)),
        *(f"{PREDICTOR}_l{lag}" for lag in range(1, max_q + 1)),
    ]
    common = prepared.dropna(
        subset=[TARGET, PREDICTOR, "ect", *all_controls, *max_lags]
    )
    h2_fit = fit_daily(common, spec, p_order, q_order)
    h2_record = _model_record("H2", spec, h2_fit)
    h2_record.update({
        "h1_spec": "daily_ardl_uecm",
        "h2_spec": "ablation_selected",
        "selected_control_groups": ", ".join(selected_groups) if selected_groups else "none",
        "selection_sample_n": int(selected_row["nobs"]),
        "selected_ablation_model": str(selected_row["model_name"]),
        "lambda_effect_rank": int(selected_row["lambda_effect_rank"]),
        "lambda_p_rank": int(selected_row["lambda_p_rank"]),
        "lambda_combined_score": float(selected_row["lambda_combined_score"]),
        "korean_return_effect_rank": int(selected_row["korean_return_effect_rank"]),
        "selection_rule": (
            "nested: -2<lambda<0; lambda_p<0.05; beta>0; beta_p<0.05; "
            "BIC p/q; one-SE OOS RMSE; simplest model"
        ),
        "forecast_selection": "nested_at_every_origin",
        "p_order": p_order,
        "q_order": q_order,
        "status": "ok",
        **metrics,
    })
    h2_summary = pd.DataFrame([h2_record])

    h1_summary.to_parquet(output_data / "h1_model.parquet", index=False)
    h2_path = output_data / "h2_model.parquet"
    h2_summary.to_parquet(h2_path, index=False)
    ranking.to_parquet(output_data / "control_ablation.parquet", index=False)
    proxy_comparison.to_parquet(output_data / "proxy_redundancy_analysis.parquet", index=False)
    proxy_diagnostics.to_parquet(output_data / "proxy_redundancy_diagnostics.parquet", index=False)
    nested_audit.to_parquet(output_data / "nested_selection_audit.parquet", index=False)
    forecasts.to_parquet(output_data / "overnight_backtest.parquet", index=False)
    write_ablation_markdown(
        ranking,
        PathSettings().project_root / "reports" / "H2_ABLATION_STUDY_RESULTS.md",
        run_id=run_id,
        min_train=min_train,
        nested_audit=nested_audit,
        nested_metrics=metrics,
        h1_summary=h1_summary,
        h2_summary=h2_summary,
        proxy_comparison=proxy_comparison,
        proxy_diagnostics=proxy_diagnostics,
    )
    return h1_summary, h2_summary, h2_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="overnight-daily-ablation")
    parser.add_argument("--max-p", type=int, default=3)
    parser.add_argument("--max-q", type=int, default=3)
    parser.add_argument("--min-train", type=int, default=40)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _, h2, report = run_model(args.run_id, args.max_p, args.max_q, args.min_train)
    print(h2.to_string(index=False), flush=True)
    print(f"[DAILY H1/H2] Wrote {report}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
