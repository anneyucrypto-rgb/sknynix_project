import numpy as np
import pandas as pd

from src.hypotheses.overnight_ablation import (
    H1_CONTROL_SETS,
    H2_SPECS,
    OvernightSpec,
    _design,
    walk_forward,
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
