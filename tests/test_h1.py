import numpy as np
import pandas as pd

from src.hypotheses.h1 import (
    bounds_test,
    equilibrium_error,
    estimate_h1,
    narayan_case_iii_bounds,
    rolling_stability,
)


def _cointegrated_frame(rows: int = 90) -> pd.DataFrame:
    rng = np.random.default_rng(2026)
    delta_k = rng.normal(0.002, 0.012, rows)
    k_level = 3.5 + np.cumsum(delta_k)
    error = np.empty(rows)
    error[0] = 0.0
    for index in range(1, rows):
        error[index] = 0.55 * error[index - 1] + rng.normal(0.0, 0.008)
    a_level = 0.7 + 0.82 * k_level + error
    return pd.DataFrame({
        "date_us": pd.date_range("2024-01-01", periods=rows).date,
        "previous_us_close": np.exp(a_level),
        "kr_fx_adjusted_close_usd": np.exp(k_level),
    })


def test_ardl_uecm_recovers_long_run_equilibrium_and_ect():
    frame = _cointegrated_frame()
    estimate = estimate_h1(frame, max_p=2, max_q=2)
    assert estimate.summary["rho_a"] < 0
    assert abs(float(estimate.summary["theta"]) - 0.82) < 0.12
    assert abs(float(estimate.summary["intercept"]) - 0.7) < 0.45
    ect = equilibrium_error(frame, estimate.summary)
    assert ect.index.equals(frame.index)
    assert np.isfinite(ect).all()


def test_bounds_inference_reports_narayan_and_bootstrap_results():
    estimate = estimate_h1(_cointegrated_frame(70), max_p=1, max_q=1)
    result = bounds_test(estimate, bootstrap_replications=19, seed=7)
    assert result["f_statistic"] > 0
    assert result["narayan_applicable"] is True
    assert result["bootstrap_successful"] == 19
    assert 0 < result["bootstrap_p_value"] <= 1


def test_narayan_bounds_are_only_used_for_supported_finite_samples():
    assert narayan_case_iii_bounds(29) is None
    assert narayan_case_iii_bounds(40)["table_n"] == 45
    assert narayan_case_iii_bounds(76) is None


def test_rolling_stability_reports_required_diagnostics():
    rolling = rolling_stability(
        _cointegrated_frame(70), window=40, step=15, max_p=1, max_q=1
    )
    assert not rolling.empty
    assert {"theta", "intercept", "residual_persistence"}.issubset(rolling)
    assert rolling["window_end"].is_monotonic_increasing

