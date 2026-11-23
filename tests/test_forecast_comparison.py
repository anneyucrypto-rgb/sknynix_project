import numpy as np
import pandas as pd

from backtest.chronos_2 import COVARIATES
from backtest.forecast_comparison import compare_common_forecasts
from src.features.chronos_2 import CHRONOS_COLUMNS


def test_chronos_daily_covariates_match_requested_variables():
    expected = (
        "kr_fx_adjusted_return",
        "vix_daily_return",
        "kr_proxy_flkr_return",
        "kr_proxy_usdkrw_return",
    )
    assert COVARIATES == expected
    assert CHRONOS_COLUMNS == ("date_us", "overnight_return", *expected)


def test_forecast_comparison_uses_common_dates_only():
    dates = pd.date_range("2026-01-01", periods=5).date
    chronos = pd.DataFrame({
        "date_us": dates,
        "forecast_log_return": np.linspace(-0.01, 0.01, 5),
        "actual_log_return": np.linspace(-0.02, 0.02, 5),
    })
    h2 = pd.DataFrame({
        "date_us": dates[1:],
        "forecast_return": np.linspace(-0.01, 0.01, 4),
        "actual_return": np.linspace(-0.01, 0.02, 4),
    })
    common, summary = compare_common_forecasts(chronos, h2)
    assert len(common) == 8
    assert summary["n_forecasts"].eq(4).all()
    assert set(common["date_us"]) == set(dates[1:])
