import numpy as np
import pandas as pd

from backtest.strategy import PremarketStrategySpec, evaluate_premarket_to_open_strategy


def _strategy_frame() -> pd.DataFrame:
    forecasts = np.r_[np.linspace(-0.01, 0.01, 10), 0.02, -0.02]
    tradable = 0.5 * forecasts
    frame = pd.DataFrame({
        "date_us": pd.date_range("2026-01-01", periods=len(forecasts)).date,
        "forecast_return": forecasts,
        "actual_return": forecasts,
        "tradable_return_0925_0930": tradable,
        "entry_0925": 100.0,
        "open_0930": 100.0 * np.exp(tradable),
        "premarket_exit_dollar_volume": 10_000_000.0,
        "opening_dollar_volume": 10_000_000.0,
    })
    for offset in range(5):
        frame[f"trade_high_m{offset}"] = 100.0
        frame[f"trade_low_m{offset}"] = 100.0
    frame.loc[10, "trade_high_m0"] = 101.0
    frame.loc[11, "trade_low_m0"] = 99.0
    return frame


def test_premarket_strategy_uses_prior_calibration_and_take_profit_orders():
    results, metrics = evaluate_premarket_to_open_strategy(
        _strategy_frame(),
        PremarketStrategySpec(
            calibration_observations=10,
            safety_buffer_bps=0,
            transaction_cost_bps_per_side=0,
            slippage_bps_per_side=0,
            impact_coefficient_bps=0,
            risk_aversion=0.001,
            minimum_volatility_bps=1,
            stop_loss_floor_bps=20,
            take_profit_floor_bps=40,
        ),
    )
    assert results.iloc[:10]["position"].eq(0).all()
    assert results.iloc[10:]["position"].tolist() == [1, -1]
    assert results.iloc[10:]["exit_reason"].str.startswith("take_profit").all()
    assert metrics["active_trades"] == 2
    assert metrics["take_profit_hits"] == 2
    assert metrics["ending_fund"] > metrics["initial_fund"]


def test_cost_hurdle_blocks_low_expected_profit():
    results, metrics = evaluate_premarket_to_open_strategy(
        _strategy_frame(),
        PremarketStrategySpec(calibration_observations=10, safety_buffer_bps=500),
    )
    assert results["position"].eq(0).all()
    assert metrics["active_trades"] == 0


def test_protective_exit_is_not_assumed_after_moo_cutoff():
    frame = _strategy_frame()
    frame.loc[10, "trade_high_m0"] = 100.0
    frame.loc[10, "trade_high_m3"] = 110.0
    results, _ = evaluate_premarket_to_open_strategy(
        frame,
        PremarketStrategySpec(
            calibration_observations=10,
            safety_buffer_bps=0,
            transaction_cost_bps_per_side=0,
            slippage_bps_per_side=0,
            impact_coefficient_bps=0,
            risk_aversion=0.001,
            minimum_volatility_bps=1,
            stop_loss_floor_bps=20,
            take_profit_floor_bps=40,
        ),
    )
    assert results.loc[10, "exit_reason"] == "moo_0930"
