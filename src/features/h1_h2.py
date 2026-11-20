"""Materialize the daily H1 and 09:25 H2 input tables."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import PROJECT_ROOT
from src.features.utils import (
    RAW_DATA,
    _parquet_tree,
    _utc,
    build_overnight_features,
    log_kr_usd,
    write_feature_table,
)


H1_FEATURE_PATH = PROJECT_ROOT / "data" / "features" / "daily" / "part-00000.parquet"
H2_FEATURE_PATH = PROJECT_ROOT / "data" / "features" / "overnight_h2" / "part-00000.parquet"


def _attach_fx_equilibrium_inputs(
    frame: pd.DataFrame, raw: Path = RAW_DATA
) -> pd.DataFrame:
    """Attach the FX inputs used by the equilibrium model."""
    fx = _utc(_parquet_tree(raw / "fx" / "USDKRW_1min"))
    fx_available = fx[["timestamp", "close"]].copy()
    fx_available["fx_available_at_utc"] = fx_available["timestamp"] + pd.Timedelta(
        minutes=1
    )
    fx_available = fx_available.rename(columns={"close": "usdkrw_cutoff"})
    frame = pd.merge_asof(
        frame.sort_values("forecast_cutoff_utc"),
        fx_available[["fx_available_at_utc", "usdkrw_cutoff"]].sort_values(
            "fx_available_at_utc"
        ),
        left_on="forecast_cutoff_utc",
        right_on="fx_available_at_utc",
        direction="backward",
        tolerance=pd.Timedelta("15min"),
    )
    frame["log_kr_usd_cutoff"] = log_kr_usd(frame["kr_close_known"], frame["usdkrw_cutoff"])
    return frame


def build_h2_features() -> pd.DataFrame:
    """Build the overnight feature panel with H2 equilibrium inputs."""

    frame = _attach_fx_equilibrium_inputs(build_overnight_features())

    frame = frame.sort_values("date_us").reset_index(drop=True)
    for lag in range(1, 4):
        frame[f"overnight_return_l{lag}"] = frame["overnight_return"].shift(lag)
    frame["valid_overnight"] = frame[
        [
            "overnight_return",
            "kr_day_return",
            "previous_us_close",
            "open_0930",
            "usdkrw_cutoff",
            "log_kr_usd_cutoff",
            "soxx_extended_return",
            "soxx_premarket_return",
            "tradable_return_0925_0930",
        ]
    ].notna().all(axis=1)
    return frame


def write_h2_features() -> pd.DataFrame:
    frame = build_h2_features()
    output = write_feature_table(frame, H2_FEATURE_PATH)
    print(f"Wrote {len(frame)} H2 feature rows to {output}")
    return frame


def main() -> None:
    from src.engine.dataset_builder import DatasetBuilder

    with DatasetBuilder() as builder:
        builder.build("daily", force=True)
    write_h2_features()


if __name__ == "__main__":
    main()
