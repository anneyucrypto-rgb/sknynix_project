from unittest.mock import patch

import numpy as np
import pandas as pd

from src.hypotheses.overnight_ablation import (
    H1_CONTROL_SETS,
    H2_SPECS,
    OvernightSpec,
    _design,
    walk_forward,
    walk_forward_h2,
)


def test_retained_design_uses_daily_return_ect_and_two_soxx_gauges():
    rows = 50
    frame = pd.DataFrame({
        "date_us": pd.date_range("2026-01-01", periods=rows).date,
        "overnight_return": np.linspace(-0.02, 0.02, rows),
        "kr_day_return": np.linspace(-0.01, 0.01, rows),
        "ect": np.linspace(0.01, -0.01, rows),
        "soxx_extended_return": 0.001,
        "soxx_premarket_return": -0.001,
    })
    spec = OvernightSpec(
        ("kr_day_return", "soxx_extended_return", "soxx_premarket_return"),
        True,
    )
    design = _design(frame, spec, p_order=0, q_order=0)
    assert list(design.columns) == [
        "const", "kr_day_return", "soxx_extended_return",
        "soxx_premarket_return", "ect",
    ]


def test_overnight_walk_forward_uses_only_prior_rows():
    rows = 55
    frame = pd.DataFrame({
        "date_us": pd.date_range("2026-01-01", periods=rows).date,
        "kr_day_return": np.linspace(-0.02, 0.02, rows),
    })
    frame["overnight_return"] = 0.5 * frame["kr_day_return"]
    spec = OvernightSpec(("kr_day_return",), False)
    forecasts, metrics = walk_forward(frame, spec, 0, 0, min_train=40)
    assert len(forecasts) == 15
    assert forecasts.iloc[0]["training_n"] == 40
    assert forecasts.iloc[-1]["training_n"] == 54
    assert metrics["rmse"] < 1e-10


def test_only_bivariate_h1_and_soxx_gauge_h2_are_registered():
    assert H1_CONTROL_SETS == {"bivariate": ()}
    assert tuple(H2_SPECS) == ("korean_ect_soxx_gauges",)
    assert H2_SPECS["korean_ect_soxx_gauges"].static_controls == (
        "kr_day_return", "soxx_extended_return", "soxx_premarket_return",
    )


def test_h2_lag_selection_and_ect_do_not_use_future_observations():
    rng = np.random.default_rng(14)
    dates = pd.date_range("2026-01-01", periods=55).date
    spec = H2_SPECS["korean_ect_soxx_gauges"]
    overnight = pd.DataFrame({
        "date_us": dates,
        "overnight_return": rng.normal(size=len(dates)),
        "previous_us_close": rng.uniform(80, 120, size=len(dates)),
        "log_kr_usd_cutoff": rng.normal(size=len(dates)),
        **{column: rng.normal(size=len(dates)) for column in spec.static_controls},
    })
    for lag in range(1, 4):
        overnight[f"overnight_return_l{lag}"] = overnight["overnight_return"].shift(lag)
    daily = pd.DataFrame({"date_us": dates, "log_adr": rng.normal(size=len(dates))})
    altered_daily = daily.copy()
    altered_daily.loc[45:, "log_adr"] += 100
    altered_overnight = overnight.assign(ect=999.0)
    altered_overnight.loc[45:, "overnight_return"] += 100

    def fit_h1(history, controls, max_lags, *, diagnostics):
        assert controls == () and not diagnostics
        return {"intercept": 0.0, "theta": history["log_adr"].mean()}, None

    with patch("src.hypotheses.overnight_ablation.fit_h1", side_effect=fit_h1):
        baseline, metrics = walk_forward_h2(overnight, daily, spec)
        altered, altered_metrics = walk_forward_h2(altered_overnight, altered_daily, spec)

    assert metrics["p_daily"] == altered_metrics["p_daily"]
    assert baseline.iloc[0]["training_n"] == 40
    assert (baseline["training_through"] < baseline["date_us"]).all()
    prior = baseline["date_us"] <= dates[45]
    np.testing.assert_allclose(
        baseline.loc[prior, "forecast_return"], altered.loc[prior, "forecast_return"]
    )
