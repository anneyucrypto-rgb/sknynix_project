from pathlib import Path

import pandas as pd

from src.dashboard.findings import (
    h2_uses_thirty_minute_artifacts,
    model_conclusions,
)


def _empty_run() -> dict[str, pd.DataFrame | Path]:
    return {
        "h1_model": pd.DataFrame(),
        "h1_bounds": pd.DataFrame(),
        "h2_models": pd.DataFrame(),
        "h2_config": pd.DataFrame(),
        "h3_model": pd.DataFrame(),
    }


def test_current_h2_schema_is_recognized():
    data = _empty_run()
    data["h2_models"] = pd.DataFrame({
        "lambda_30": [-0.02],
        "lambda_30_p_value": [0.01],
    })
    data["h2_config"] = pd.DataFrame({"interval_minutes": [30]})

    assert h2_uses_thirty_minute_artifacts(data)


def test_legacy_five_minute_artifacts_do_not_crash_dashboard():
    data = _empty_run()
    data["h2_models"] = pd.DataFrame({
        "reference": ["dynamic"],
        "anchor": ["liquid"],
        "lambda_5": [0.001],
        "lambda_5_p_value": [0.50],
    })
    data["h2_config"] = pd.DataFrame({"interval_minutes": [5]})

    conclusions = model_conclusions(data)

    assert not h2_uses_thirty_minute_artifacts(data)
    assert conclusions["h2"]["status"] == "Rebuild required"
