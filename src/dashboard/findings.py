"""Load canonical H1--H3 artifacts and compute shared model diagnostics."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant

from src.config import PROJECT_ROOT
from src.features.h1_h2 import H1_FEATURE_PATH
from src.hypotheses.h1 import estimate_h1
from src.hypotheses.h2 import fit_intraday_ecm
from src.hypotheses.h3 import fit_h3_model


RUNS = {
    "Current canonical estimates": {
        "h1": PROJECT_ROOT / "data/features/experiments/run_id=h1-daily-ardl",
        "h2": PROJECT_ROOT / "data/features/experiments/run_id=h2-intraday-convergence",
        "h3": PROJECT_ROOT / "data/features/experiments/run_id=h3-next-korean-session",
    },
}


def _read(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.is_file() else pd.DataFrame()


def load_run(label: str) -> dict[str, pd.DataFrame | Path]:
    paths = RUNS[label]
    return {
        "h1_model": _read(paths["h1"] / "h1_model.parquet"),
        "h1_bounds": _read(paths["h1"] / "h1_bounds.parquet"),
        "h1_ect": _read(paths["h1"] / "h1_ect.parquet"),
        "h1_rolling": _read(paths["h1"] / "h1_rolling_stability.parquet"),
        "h2_models": _read(paths["h2"] / "h2_ecm_models.parquet"),
        "h2_convergence": _read(paths["h2"] / "h2_convergence_summary.parquet"),
        "h2_panel": _read(paths["h2"] / "h2_intraday_panel.parquet"),
        "h3_model": _read(paths["h3"] / "h3_model.parquet"),
        "h3_coefficients": _read(paths["h3"] / "h3_coefficients.parquet"),
        "h3_panel": _read(paths["h3"] / "h3_panel.parquet"),
        "h1_report": PROJECT_ROOT / "design/h1.md",
        "h2_report": PROJECT_ROOT / "design/h2.md",
        "h3_report": PROJECT_ROOT / "design/h3.md",
    }


def model_conclusions(data: dict[str, pd.DataFrame | Path]) -> dict[str, dict[str, str]]:
    h1_model = data["h1_model"]
    h1_bounds = data["h1_bounds"]
    h2 = data["h2_models"]
    h3 = data["h3_model"]
    output = {
        "h1": {"status": "Not estimated", "detail": "Run the canonical H1 model."},
        "h2": {"status": "Not estimated", "detail": "Run the canonical H2 model."},
        "h3": {"status": "Not estimated", "detail": "Run the canonical H3 model."},
    }
    if isinstance(h1_model, pd.DataFrame) and not h1_model.empty:
        decision = ""
        if isinstance(h1_bounds, pd.DataFrame) and not h1_bounds.empty:
            decision = str(h1_bounds.iloc[0].get("bootstrap_decision_05", ""))
        theta = float(h1_model.iloc[0]["theta"])
        output["h1"] = {
            "status": "Cointegrated" if decision == "cointegrated" else "Inconclusive",
            "detail": f"Long-run slope θ = {theta:.3f}; bootstrap decision: {decision or 'unavailable'}.",
        }
    if isinstance(h2, pd.DataFrame) and not h2.empty:
        preferred = h2.loc[
            h2["reference"].eq("dynamic") & h2["anchor"].eq("liquid")
        ]
        row = (preferred if not preferred.empty else h2).iloc[0]
        stable = bool(row["monotonic_stable"])
        output["h2"] = {
            "status": "Convergent" if stable else "No convergence detected",
            "detail": f"Thirty-minute λ = {float(row['lambda_30']):.4f}; p = {float(row['lambda_30_p_value']):.3f}.",
        }
    if isinstance(h3, pd.DataFrame) and not h3.empty:
        row = h3.iloc[0]
        supported = bool(row["positive_at_5pct"])
        output["h3"] = {
            "status": "Supported" if supported else "Not supported",
            "detail": f"Korean adjustment λK = {float(row['lambda_k']):.3f}; one-sided p = {float(row['lambda_k_p_value_positive']):.4f}.",
        }
    return output


def _interpret(p_value: float) -> str:
    if not np.isfinite(p_value):
        return "Descriptive only"
    return "Potential misspecification" if p_value < 0.05 else "No rejection at 5%"


def _auxiliary_lm(target: np.ndarray, regressors: np.ndarray, df: int) -> tuple[float, float]:
    auxiliary = OLS(target, regressors).fit()
    statistic = len(target) * float(auxiliary.rsquared)
    return statistic, float(stats.chi2.sf(statistic, max(1, df)))


def _diagnostics_for_fit(name: str, fit: object) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    tests: list[tuple[str, object]] = []
    residual = np.asarray(fit.resid, dtype=float)
    exog = np.asarray(fit.model.exog, dtype=float)
    fitted = np.asarray(fit.fittedvalues, dtype=float)
    nobs, nparams = exog.shape
    nlags = min(5, max(1, nobs // 10))
    try:
        lagged = np.column_stack([
            residual[nlags - lag:nobs - lag] for lag in range(1, nlags + 1)
        ])
        statistic, p_value = _auxiliary_lm(
            residual[nlags:], np.column_stack([exog[nlags:], lagged]), nlags
        )
        tests.append(("Breusch–Godfrey", (statistic, p_value)))
    except Exception:
        tests.append(("Breusch–Godfrey", (math.nan, math.nan)))
    try:
        statistic, p_value = _auxiliary_lm(residual**2, exog, max(1, nparams - 1))
        tests.append(("Breusch–Pagan", (statistic, p_value)))
    except Exception:
        tests.append(("Breusch–Pagan", (math.nan, math.nan)))
    try:
        augmented = np.column_stack([exog, fitted**2])
        reset = OLS(np.asarray(fit.model.endog, dtype=float), augmented).fit()
        restricted_ssr = float(np.sum(residual**2))
        unrestricted_ssr = float(np.sum(reset.resid**2))
        statistic = ((restricted_ssr - unrestricted_ssr) / 1) / (
            unrestricted_ssr / max(1, nobs - nparams - 1)
        )
        tests.append(("Ramsey RESET", (statistic, float(stats.f.sf(statistic, 1, nobs - nparams - 1)))))
    except Exception:
        tests.append(("Ramsey RESET", (math.nan, math.nan)))
    try:
        result = stats.jarque_bera(residual)
        tests.append(("Jarque–Bera", (float(result.statistic), float(result.pvalue))))
    except Exception:
        tests.append(("Jarque–Bera", (math.nan, math.nan)))
    try:
        squared = residual**2
        lagged = np.column_stack([
            squared[nlags - lag:nobs - lag] for lag in range(1, nlags + 1)
        ])
        statistic, p_value = _auxiliary_lm(
            squared[nlags:], add_constant(lagged, has_constant="add"), nlags
        )
        tests.append(("ARCH-LM", (statistic, p_value)))
    except Exception:
        tests.append(("ARCH-LM", (math.nan, math.nan)))
    try:
        centered = residual - np.mean(residual)
        scale = np.std(centered, ddof=nparams)
        statistic = float(np.max(np.abs(np.cumsum(centered))) / (scale * np.sqrt(nobs)))
        p_value = float(stats.kstwobign.sf(statistic))
        tests.append(("CUSUM", (statistic, p_value)))
    except Exception:
        tests.append(("CUSUM", (math.nan, math.nan)))

    centered = residual - np.nanmean(residual)
    squares = centered**2
    cusumsq = (
        float(np.nanmax(np.abs(np.nancumsum(squares) / np.nansum(squares) - np.arange(1, len(squares) + 1) / len(squares))))
        if len(squares) and np.nansum(squares) > 0 else math.nan
    )
    tests.append(("CUSUMSQ", (cusumsq, math.nan)))

    for diagnostic, (statistic, p_value) in tests:
        rows.append({
            "Model": name,
            "Diagnostic": diagnostic,
            "Statistic": float(statistic),
            "p-value": float(p_value),
            "Interpretation": _interpret(float(p_value)),
        })
    return rows


def shared_diagnostics(data: dict[str, pd.DataFrame | Path]) -> pd.DataFrame:
    """Recreate representative fits and apply a common diagnostic battery."""
    rows: list[dict[str, object]] = []
    try:
        daily = pd.read_parquet(H1_FEATURE_PATH)
        rows.extend(_diagnostics_for_fit("H1 daily UECM", estimate_h1(daily).fit))
    except Exception:
        pass
    try:
        h2_panel = data["h2_panel"]
        if isinstance(h2_panel, pd.DataFrame) and not h2_panel.empty:
            _, fit = fit_intraday_ecm(
                h2_panel, reference="dynamic", anchor="liquid", session="regular"
            )
            rows.extend(_diagnostics_for_fit("H2 dynamic/liquid", fit))
    except Exception:
        pass
    try:
        h3_panel = data["h3_panel"]
        if isinstance(h3_panel, pd.DataFrame) and not h3_panel.empty:
            _, _, fit = fit_h3_model(h3_panel)
            rows.extend(_diagnostics_for_fit("H3 U.S.→Korea", fit))
    except Exception:
        pass
    return pd.DataFrame(rows)
