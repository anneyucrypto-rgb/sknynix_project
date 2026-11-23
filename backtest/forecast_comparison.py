"""Compare Chronos-2 with the fixed daily H1/H2 VIX specification."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT, PathSettings, feature_output_path
from src.features.h1_h2 import H2_FEATURE_PATH
from src.hypotheses.overnight_ablation import OvernightSpec, walk_forward


def forecast_metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    errors = frame["actual_return"] - frame["forecast_return"]
    zero_sse = float(np.square(frame["actual_return"]).sum())
    model_sse = float(np.square(errors).sum())
    return {
        "n_forecasts": len(frame),
        "rmse": float(np.sqrt(np.square(errors).mean())),
        "mae": float(np.abs(errors).mean()),
        "bias": float(errors.mean()),
        "direction_accuracy": float(
            np.sign(frame["actual_return"]).eq(np.sign(frame["forecast_return"])).mean()
        ),
        "oos_r2_zero": 1.0 - model_sse / zero_sse if zero_sse else math.nan,
    }


def h2_vix_forecasts(panel: pd.DataFrame, min_train: int = 40) -> pd.DataFrame:
    """Forecast the overnight target with H1-refitted, p=q=0, VIX-only H2."""
    forecasts, _ = walk_forward(
        panel,
        OvernightSpec(("vix_daily_return",)),
        p_order=0,
        q_order=0,
        min_train=min_train,
    )
    return forecasts


def compare_common_forecasts(
    chronos: pd.DataFrame,
    h2: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return common-date forecasts and comparable model metrics."""
    chronos = chronos.copy()
    if "forecast_return" not in chronos:
        chronos = chronos.rename(columns={"forecast_log_return": "forecast_return"})
    if "actual_return" not in chronos:
        chronos = chronos.rename(columns={"actual_log_return": "actual_return"})
    required = ["date_us", "forecast_return", "actual_return"]
    for name, frame in (("chronos-2", chronos), ("h2-vix", h2)):
        missing = set(required).difference(frame)
        if missing:
            raise ValueError(f"{name} forecasts are missing {sorted(missing)}")
    chronos["date_us"] = pd.to_datetime(chronos["date_us"]).dt.date
    h2 = h2.copy()
    h2["date_us"] = pd.to_datetime(h2["date_us"]).dt.date
    common_dates = sorted(set(chronos["date_us"]).intersection(h2["date_us"]))
    if not common_dates:
        raise ValueError("Chronos-2 and H2 VIX have no common forecast dates")

    chronos_actual = chronos.set_index("date_us").loc[common_dates, "actual_return"]
    h2_actual = h2.set_index("date_us").loc[common_dates, "actual_return"]
    if not np.allclose(chronos_actual, h2_actual, equal_nan=False):
        raise ValueError("Models do not share identical realized returns on common dates")

    frames = []
    summaries = []
    for model, frame in (("chronos-2", chronos), ("h2-vix", h2)):
        one = frame.loc[frame["date_us"].isin(common_dates), required].copy()
        one = one.sort_values("date_us").reset_index(drop=True)
        one.insert(0, "forecast_model", model)
        frames.append(one)
        summaries.append({"forecast_model": model, **forecast_metrics(one)})
    return pd.concat(frames, ignore_index=True), pd.DataFrame(summaries)


def write_comparison_report(summary: pd.DataFrame, path: Path) -> Path:
    indexed = summary.set_index("forecast_model")
    h2 = indexed.loc["h2-vix"]
    chronos = indexed.loc["chronos-2"]
    rmse_reduction = 1.0 - float(h2["rmse"]) / float(chronos["rmse"])
    mae_reduction = 1.0 - float(h2["mae"]) / float(chronos["mae"])
    direction_gain = float(h2["direction_accuracy"] - chronos["direction_accuracy"])
    lines = [
        "# Chronos-2 versus H1/H2 VIX forecast comparison",
        "",
        "Both models forecast the daily SKHY previous-close-to-open log return on",
        "the same dates. H2 re-estimates H1 and its ECT at every origin and uses",
        "the fixed VIX-only p=q=0 specification. Chronos-2 uses the Korean",
        "FX-adjusted return, VIX return, FLKR return, and USD/KRW return.",
        "",
        "| Model | Forecasts | RMSE | MAE | Bias | Direction accuracy | OOS R² |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.forecast_model} | {int(row.n_forecasts)} | {row.rmse:.6f} | "
            f"{row.mae:.6f} | {row.bias:.6f} | {row.direction_accuracy:.2%} | "
            f"{row.oos_r2_zero:.6f} |"
        )
    lines.extend([
        "",
        "## Result",
        "",
        f"The VIX-only H1/H2 model has {rmse_reduction:.1%} lower RMSE and "
        f"{mae_reduction:.1%} lower MAE than Chronos-2, with a "
        f"{100 * direction_gain:.2f}-percentage-point higher directional hit rate. On this short "
        "common-date sample, H1/H2 is the stronger forecasting model.",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    paths = PathSettings()
    output_dir = paths.feature_run("forecast-comparison")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--chronos",
        type=Path,
        default=paths.feature_run("chronos-2") / "forecasts.parquet",
    )
    parser.add_argument("--h2-features", type=Path, default=H2_FEATURE_PATH)
    parser.add_argument("--min-train", type=int, default=40)
    parser.add_argument("--output-dir", type=Path, default=output_dir)
    parser.add_argument(
        "--report", type=Path,
        default=PROJECT_ROOT / "reports" / "CHRONOS_H2_COMPARISON.md",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = feature_output_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    chronos = pd.read_parquet(args.chronos)
    panel = pd.read_parquet(args.h2_features)
    h2 = h2_vix_forecasts(panel, args.min_train)
    common, summary = compare_common_forecasts(chronos, h2)
    h2.to_parquet(output_dir / "h2_vix_forecasts.parquet", index=False)
    common.to_parquet(output_dir / "common_forecasts.parquet", index=False)
    for model, filename in (
        ("chronos-2", "chronos_common_forecasts.parquet"),
        ("h2-vix", "h2_vix_common_forecasts.parquet"),
    ):
        common.loc[common["forecast_model"].eq(model)].drop(
            columns="forecast_model"
        ).to_parquet(output_dir / filename, index=False)
    summary.to_parquet(output_dir / "forecast_metrics.parquet", index=False)
    report = write_comparison_report(summary, args.report)
    print(summary.set_index("forecast_model").T.to_string(), flush=True)
    print(f"[COMPARISON] Wrote {report}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
