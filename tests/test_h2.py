import numpy as np
import pandas as pd

from src.features.utils import NY
from src.hypotheses.h2 import (
    ECM_CONTROLS,
    H2Config,
    _interval_bars,
    _liquidity_anchors,
    _market_returns,
    convergence_episodes,
    convergence_summary,
    fit_intraday_ecm,
    kaplan_meier,
)


def _ecm_panel(days: int = 12, bars: int = 12) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows = []
    for day in pd.date_range("2026-09-01", periods=days, freq="B"):
        starts = pd.date_range(
            pd.Timestamp(f"{day.date()} 09:30", tz=NY), periods=bars, freq="30min"
        ).tz_convert("UTC")
        controls = rng.normal(scale=0.01, size=(bars, len(ECM_CONTROLS)))
        gap = np.empty(bars)
        gap[0] = rng.normal(scale=0.05)
        for index in range(1, bars):
            gap[index] = (
                0.72 * gap[index - 1]
                + 0.08 * controls[index - 1, 0]
                + rng.normal(scale=0.004)
            )
        for index, start in enumerate(starts):
            row = {
                "date_us": day.date(),
                "bar_start_utc": start,
                "observation_ts_utc": start + pd.Timedelta(minutes=30),
                "session": "regular",
                "g_static": gap[index],
                "g_dynamic": gap[index] * 0.9,
                "fixed_clock_ts_utc": starts[0],
                "liquidity_anchor_ts_utc": starts[0] + pd.Timedelta(minutes=60),
                "convergence_band": 0.01,
            }
            row.update(dict(zip(ECM_CONTROLS, controls[index], strict=True)))
            rows.append(row)
    return pd.DataFrame(rows)


def test_intraday_ecm_recovers_stable_thirty_minute_adjustment():
    summary, _ = fit_intraday_ecm(
        _ecm_panel(), reference="static", anchor="fixed", session="regular"
    )
    assert -1.0 < summary["lambda_30"] < 0.0
    assert summary["monotonic_stable"] is True
    assert summary["half_life_minutes"] > 0
    assert summary["t90_minutes"] > summary["half_life_minutes"]


def test_two_bar_liquidity_confirmation_cannot_anchor_before_1030():
    date = pd.Timestamp("2026-09-01")
    starts = pd.date_range(
        pd.Timestamp(f"{date.date()} 09:30", tz=NY), periods=3, freq="30min"
    ).tz_convert("UTC")
    panel = pd.DataFrame({
        "date_us": date.date(),
        "bar_start_utc": starts,
        "observation_ts_utc": starts + pd.Timedelta(minutes=30),
        "session": "regular",
        "liquidity_score": [1.2, 1.1, 1.3],
    })
    anchor = _liquidity_anchors(panel, H2Config()).iloc[0]["liquidity_anchor_ts_utc"]
    assert pd.Timestamp(anchor).tz_convert(NY).strftime("%H:%M") == "10:30"


def test_thirty_minute_market_return_reference_uses_0930_open():
    starts = pd.date_range(
        pd.Timestamp("2026-09-01 09:30", tz=NY), periods=2, freq="30min"
    ).tz_convert("UTC")
    panel = pd.DataFrame({
        "date_us": [pd.Timestamp("2026-09-01").date()] * 2,
        "bar_start_utc": starts,
        "soxx_open": [100.0, 101.0],
        "soxx_close": [101.0, 102.0],
    })

    result = _market_returns(panel, "SOXX")

    assert np.isclose(result.loc[0, "soxx_return_from_anchor"], np.log(101.0 / 100.0))
    assert np.isclose(result.loc[1, "soxx_return_from_anchor"], np.log(102.0 / 100.0))


def test_convergence_episodes_keep_nonconvergence_as_right_censored():
    panel = _ecm_panel(days=2, bars=5)
    dates = sorted(panel["date_us"].unique())
    panel.loc[panel["date_us"].eq(dates[0]), "g_static"] = [0.08, 0.03, 0.005, 0.004, 0.003]
    panel.loc[panel["date_us"].eq(dates[1]), "g_static"] = 0.08
    episodes = convergence_episodes(
        panel, reference="static", anchor="liquid", session="regular"
    )
    first = episodes.loc[episodes["date_us"].eq(dates[0])].iloc[0]
    second = episodes.loc[episodes["date_us"].eq(dates[1])].iloc[0]
    assert first["event_observed"]
    assert first["duration_liquid_minutes"] == 60.0
    assert second["right_censored"]

    curve = kaplan_meier(episodes)
    summary = convergence_summary(episodes)
    assert not curve.empty
    assert summary["eligible_sessions"] == 2
    assert summary["converged_sessions"] == 1
    assert summary["no_convergence_rate"] == 0.5


def test_missing_reference_path_is_not_counted_as_censored_session():
    panel = _ecm_panel(days=2, bars=5)
    dates = sorted(panel["date_us"].unique())
    panel.loc[panel["date_us"].eq(dates[0]), "g_dynamic"] = np.nan
    panel.loc[panel["date_us"].eq(dates[1]), "g_dynamic"] = 0.08

    episodes = convergence_episodes(
        panel, reference="dynamic", anchor="fixed", session="regular"
    )
    missing = episodes.loc[episodes["date_us"].eq(dates[0])].iloc[0]
    summary = convergence_summary(episodes)

    assert not missing["eligible"]
    assert not missing["right_censored"]
    assert summary["eligible_sessions"] == 1


def test_source_bars_are_aggregated_into_true_thirty_minute_intervals(tmp_path):
    source = tmp_path / "us" / "market" / "SKHY_5min"
    source.mkdir(parents=True)
    starts = pd.date_range("2026-09-01 13:30", periods=6, freq="5min", tz="UTC")
    frame = pd.DataFrame({
        "timestamp": starts,
        "open": np.arange(100.0, 106.0),
        "close": np.arange(100.5, 106.5),
        "volume": np.arange(1.0, 7.0),
        "vwap": np.arange(100.25, 106.25),
        "transactions": np.arange(10, 16),
    })
    frame.to_parquet(source / "part-00000.parquet", index=False)

    result = _interval_bars(tmp_path, "SKHY", H2Config())

    assert len(result) == 1
    assert result.iloc[0]["skhy_open"] == 100.0
    assert result.iloc[0]["skhy_close"] == 105.5
    assert result.iloc[0]["skhy_volume"] == 21.0

