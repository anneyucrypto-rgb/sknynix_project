import numpy as np
import pandas as pd

from src.features.h1_h2 import CONTROL_GROUPS, PREDICTOR
from src.hypotheses.overnight_ablation import (
    OvernightSpec,
    _design,
    analyze_kr_proxy_redundancy,
    nested_walk_forward_selection,
    prepare_ecm_panel,
    build_full_sample_comparison,
    walk_forward,
)


def _frame(rows: int = 70) -> pd.DataFrame:
    rng = np.random.default_rng(14)
    kr_return = rng.normal(scale=0.01, size=rows)
    log_kr = 4.0 + np.cumsum(kr_return)
    ect = np.empty(rows)
    ect[0] = 0.0
    for index in range(1, rows):
        ect[index] = 0.65 * ect[index - 1] + rng.normal(scale=0.01)
    log_us = 1.2 + 0.8 * log_kr + ect
    frame = pd.DataFrame({
        "date_us": pd.date_range("2026-01-01", periods=rows).date,
        "previous_us_close": np.exp(log_us),
        "kr_fx_adjusted_close_usd": np.exp(log_kr),
        PREDICTOR: kr_return,
    })
    for columns in CONTROL_GROUPS.values():
        for column in columns:
            frame[column] = rng.normal(scale=0.01, size=rows)
    frame["kr_proxy_ewy_return"] = rng.normal(scale=0.01, size=rows)
    frame["overnight_return"] = (
        -0.4 * ect + 0.5 * kr_return + rng.normal(scale=0.002, size=rows)
    )
    return frame


def test_h1_and_h2_design_contains_pre_response_ect():
    frame = prepare_ecm_panel(_frame())
    design = _design(frame.dropna(), OvernightSpec())
    assert list(design.columns) == ["const", PREDICTOR, "ect"]


def test_daily_walk_forward_uses_only_prior_rows():
    frame = _frame(60)
    forecasts, metrics = walk_forward(frame, OvernightSpec(), min_train=40)
    assert len(forecasts) == 20
    assert forecasts.iloc[0]["training_n"] == 40
    assert forecasts.iloc[-1]["training_n"] == 59
    assert (forecasts["training_through"] < forecasts["date_us"]).all()
    assert np.isfinite(metrics["rmse"])


def test_design_supports_daily_p_and_q_orders():
    frame = prepare_ecm_panel(_frame(), max_p=2, max_q=2).dropna()
    design = _design(frame, OvernightSpec(), p_order=2, q_order=1)
    assert "overnight_return_l2" in design
    assert f"{PREDICTOR}_l1" in design
    assert f"{PREDICTOR}_l2" not in design


def test_lambda_ranking_includes_soxx_orders_and_negative_ect_lambda():
    ranking = build_full_sample_comparison(_frame(), min_train=40, max_p=2, max_q=2)
    assert "add_soxx" in set(ranking["model_name"])
    assert "all_controls" in set(ranking["model_name"])
    assert "all_except_soxx" in set(ranking["model_name"])
    assert ranking["h2_rank"].tolist() == list(range(1, len(ranking) + 1))
    assert ranking["nobs"].nunique() == 1
    assert ranking["p_order"].between(0, 2).all()
    assert ranking["q_order"].between(0, 2).all()
    assert (ranking["lambda"] < 0).all()
    assert (ranking["lambda_p_value"] < 0.05).all()
    assert ranking["abs_korean_return_coefficient"].is_monotonic_decreasing


def test_nested_selection_does_not_use_future_rows():
    frame = _frame(65)
    groups = {name: CONTROL_GROUPS[name] for name in ("soxx", "vix")}
    baseline, _, audit = nested_walk_forward_selection(
        frame, groups, min_train=40, inner_min_train=30, max_p=1, max_q=1
    )
    altered = frame.copy()
    altered.loc[55, "overnight_return"] += 0.1
    changed, _, _ = nested_walk_forward_selection(
        altered, groups, min_train=40, inner_min_train=30, max_p=1, max_q=1
    )
    through_first_changed_date = baseline["date_us"] <= frame.loc[55, "date_us"]
    np.testing.assert_allclose(
        baseline.loc[through_first_changed_date, "forecast_return"],
        changed.loc[through_first_changed_date, "forecast_return"],
    )
    assert (baseline["training_through"] < baseline["date_us"]).all()
    assert audit.groupby("date_us")["selected"].sum().eq(1).all()


def test_proxy_redundancy_analysis_compares_all_nonempty_subsets():
    comparison, diagnostics = analyze_kr_proxy_redundancy(_frame(), min_train=40)
    assert len(comparison) == 7
    assert "FLKR + USDKRW" in set(comparison["proxy_set"])
    assert diagnostics.iloc[0]["vif__kr_proxy_ewy_return"] > 1
    assert comparison["proxy_rank"].tolist() == list(range(1, 8))
