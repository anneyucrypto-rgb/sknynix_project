"""Walk-forward backtests for the retained SKHY forecast model."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def run_overnight_backtest(
    project_root: Path,
    *,
    run_id: str = "overnight-best",
    h1_spec: str = "bivariate",
    h2_spec: str = "korean_ect_soxx_gauges",
    min_train: int = 40,
) -> tuple[pd.DataFrame, dict[str, float | int | str]]:
    """Re-estimate the retained overnight equation on expanding windows."""

    from src.hypotheses.overnight_ablation import (
        H2_SPECS,
        common_comparison_dates,
        walk_forward,
    )

    report_dir = project_root / "reports" / f"run_id={run_id}"
    data_dir = project_root / "data" / "features" / "experiments" / f"run_id={run_id}"
    h1 = pd.read_parquet(report_dir / "h1_model.parquet")
    h2 = pd.read_parquet(report_dir / "h2_model.parquet")
    h1_row = h1.loc[h1["h1_spec"].eq(h1_spec) & h1["status"].eq("ok")].iloc[0]
    h2_row = h2.loc[
        h2["h1_spec"].eq(h1_spec)
        & h2["h2_spec"].eq(h2_spec)
        & h2["status"].eq("ok")
    ].iloc[0]
    overnight = pd.read_parquet(data_dir / "overnight.parquet")
    overnight["ect"] = (
        np.log(overnight["previous_us_close"])
        - float(h1_row["intercept"])
        - float(h1_row["theta"]) * overnight["log_kr_usd_cutoff"]
    )
    dates = common_comparison_dates(overnight, h2)
    forecasts, forecast_metrics = walk_forward(
        overnight.loc[overnight["date_us"].isin(dates)],
        H2_SPECS[h2_spec],
        int(h2_row["p_daily"]),
        int(h2_row["q_five_minute"]),
        min_train=min_train,
    )
    metrics: dict[str, float | int | str] = {
        "model": "overnight",
        "h1_spec": h1_spec,
        "h2_spec": h2_spec,
        "p_daily": int(h2_row["p_daily"]),
        "q_five_minute": int(h2_row["q_five_minute"]),
        "common_comparison_n": len(dates),
        "initial_training_n": min_train,
        **forecast_metrics,
    }
    return forecasts, metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="overnight-best")
    parser.add_argument("--h1-spec", default="bivariate")
    parser.add_argument("--h2-spec", default="korean_ect_soxx_gauges")
    parser.add_argument("--min-train", type=int, default=40)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    forecasts, metrics = run_overnight_backtest(
        project_root,
        run_id=args.run_id,
        h1_spec=args.h1_spec,
        h2_spec=args.h2_spec,
        min_train=args.min_train,
    )
    report_dir = project_root / "reports" / f"run_id={args.run_id}"
    output = args.output or report_dir / "overnight_backtest.parquet"
    output.parent.mkdir(parents=True, exist_ok=True)
    forecasts.to_parquet(output, index=False)
    pd.DataFrame([metrics]).to_parquet(
        output.with_name(output.stem + "_summary.parquet"), index=False
    )
    print(pd.Series(metrics).to_string(), flush=True)
    print(f"[BACKTEST] Wrote {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
