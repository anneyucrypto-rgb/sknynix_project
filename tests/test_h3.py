import numpy as np
import pandas as pd

from src.features.utils import NY, SEOUL
from src.hypotheses.h3 import DEFAULT_CONTROLS, fit_h3_model, pair_next_korean_session


def test_pairing_uses_strictly_future_korean_session_and_last_us_close():
    us_close = pd.to_datetime([
        pd.Timestamp("2026-09-04 16:00", tz=NY),
        pd.Timestamp("2026-09-07 16:00", tz=NY),
    ]).tz_convert("UTC")
    us = pd.DataFrame({"date_us": [pd.Timestamp("2026-09-04").date(), pd.Timestamp("2026-09-07").date()], "us_close_utc": us_close})
    kr_dates = pd.to_datetime(["2026-09-04", "2026-09-08"])
    korean = pd.DataFrame({
        "kr_session_date": kr_dates.date,
        "kr_available_utc": pd.to_datetime(kr_dates.astype(str) + " 15:30").tz_localize(SEOUL).tz_convert("UTC"),
        "kr_close": [100.0, 110.0],
    })

    paired = pair_next_korean_session(us, korean)

    assert paired["kr_target_date"].tolist() == [pd.Timestamp("2026-09-08").date()] * 2
    assert paired["last_us_close_before_target"].tolist() == [False, True]
    assert np.allclose(paired["kr_next_return"], np.log(1.1))


def test_h3_model_recovers_positive_korean_error_correction():
    rng = np.random.default_rng(21)
    nobs = 120
    ect = rng.normal(scale=0.04, size=nobs)
    adr = rng.normal(scale=0.02, size=nobs)
    controls = rng.normal(scale=0.01, size=(nobs, len(DEFAULT_CONTROLS)))
    target = 0.55 * ect - 0.10 * adr + 0.20 * controls[:, 0] + rng.normal(scale=0.008, size=nobs)
    panel = pd.DataFrame({
        "date_us": pd.date_range("2026-01-02", periods=nobs, freq="B").date,
        "kr_target_date": pd.date_range("2026-01-05", periods=nobs, freq="B").date,
        "kr_next_return": target,
        "ect_us_close": ect,
        "adr_return": adr,
        "valid_h3": True,
    })
    for index, control in enumerate(DEFAULT_CONTROLS):
        panel[control] = controls[:, index]

    summary, _, _ = fit_h3_model(panel)

    assert summary["lambda_k"] > 0.0
    assert summary["positive_at_5pct"] is True
    assert abs(summary["lambda_k"] - 0.55) < 0.10
