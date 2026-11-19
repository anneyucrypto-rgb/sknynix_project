"""Walk-forward SKHY close-to-open forecasts and simulated 09:30 ET entries."""

from __future__ import annotations

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory
import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config import PROJECT_ROOT
from src.hypotheses.overnight_ablation import H2_SPECS, build_overnight_dataset, fit_h1


TARGET = "overnight_return"
COVARIATES = ("ect", *H2_SPECS["korean_ect_soxx_gauges"].static_controls)
ECT_INPUTS = ["previous_us_close", "log_kr_usd_cutoff"]
BASE_MODEL_ID = "amazon/chronos-2"
BASE_MODEL_DIR = PROJECT_ROOT / ".model_cache" / "chronos-2"


def load_base_model(pipeline_class, checkpoint: Path, device: str):
    if checkpoint.exists():
        print(f"Loading Chronos-2 from {checkpoint}", flush=True)
        return pipeline_class.from_pretrained(
            str(checkpoint), device_map=device, local_files_only=True
        )

    print(f"Preparing local Chronos-2 checkpoint at {checkpoint}", flush=True)
    with warnings.catch_warnings():
        # Hugging Face can cache the initial download without symlinks on Windows.
        warnings.filterwarnings(
            "ignore",
            message=r".*cache-system uses symlinks by default",
            category=UserWarning,
            module=r"huggingface_hub\.file_download",
        )
        pipeline = pipeline_class.from_pretrained(BASE_MODEL_ID, device_map=device)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    # Publish the checkpoint only after the weights and config have been saved.
    with TemporaryDirectory(prefix="chronos-2-", dir=checkpoint.parent) as staging:
        saved = Path(staging) / "checkpoint"
        pipeline.save_pretrained(saved)
        try:
            saved.rename(checkpoint)
        except OSError:
            if not checkpoint.is_dir():
                raise
            # Another run published its checkpoint while this one was saving.
    print(f"Local Chronos-2 checkpoint ready at {checkpoint}", flush=True)
    return pipeline


def chronos_input(history: pd.DataFrame, current: pd.Series | None = None) -> dict:
    return {
        "target": history[TARGET].to_numpy(dtype=np.float32),
        "past_covariates": {
            name: history[name].to_numpy(dtype=np.float32) for name in COVARIATES
        },
        # During fitting these keys tell Chronos to take known covariates from
        # the label step of each historical training window.
        "future_covariates": {
            name: np.array([np.nan if current is None else current[name]], dtype=np.float32)
            for name in COVARIATES
        },
    }


def walk_forward(
    overnight: pd.DataFrame,
    daily: pd.DataFrame,
    base_pipeline,
    *,
    min_train: int,
    steps: int,
    max_forecasts: int | None = None,
) -> pd.DataFrame:
    required = [*ECT_INPUTS, *COVARIATES[1:]]
    panel = overnight.sort_values("date_us").replace([np.inf, -np.inf], np.nan)
    panel = panel.dropna(subset=required)
    panel = panel.loc[panel["previous_us_close"].gt(0)].reset_index(drop=True)
    daily = daily.sort_values("date_us")
    daily_dates = pd.to_datetime(daily["date_us"])
    if panel["date_us"].duplicated().any() or daily_dates.duplicated().any():
        raise ValueError("Expected one observation per US session in each dataset")

    prior_observations = panel[TARGET].notna().cumsum().shift(fill_value=0)
    forecast_indices = panel.index[prior_observations.ge(min_train)][:max_forecasts]
    if forecast_indices.empty:
        raise ValueError("No forecast dates with enough complete prior observations")

    records = []
    with (
        TemporaryDirectory(prefix="skhy-chronos-") as work_dir,
        tqdm(total=len(forecast_indices), desc="Chronos-2 test forecasts", unit="forecast") as progress,
    ):
        for index in forecast_indices:
            current = panel.iloc[index].copy()
            history = panel.iloc[:index].dropna(subset=[TARGET]).copy()
            status = {"date": str(current["date_us"]), "train_n": len(history)}
            progress.set_postfix(status, phase="training")

            h1_history = daily.loc[daily_dates.lt(pd.Timestamp(current["date_us"]))]
            h1, _ = fit_h1(h1_history, (), max_lags=3)
            # Apply the same prior-only H1 vintage to training and inference.
            for rows in (history, current):
                rows["ect"] = (
                    np.log(rows["previous_us_close"])
                    - h1["intercept"]
                    - h1["theta"] * rows["log_kr_usd_cutoff"]
                )
            if not np.isfinite(history["ect"]).all() or not np.isfinite(current["ect"]):
                raise ValueError(f"Non-finite ECT at {current['date_us']}")

            # fit() copies the base checkpoint. Each origin gets a fresh LoRA
            # fit on every completed observation available at its cutoff.
            fitted = base_pipeline.fit(
                inputs=[chronos_input(history)],
                prediction_length=1,
                finetune_mode="lora",
                learning_rate=1e-5,
                num_steps=steps,
                batch_size=16,
                min_past=8,
                output_dir=Path(work_dir),
                save_strategy="no",
                report_to=[],
                seed=42,
                disable_tqdm=True,
                remove_printer_callback=True,
            )
            progress.set_postfix(status, phase="testing")
            quantiles, _ = fitted.predict_quantiles(
                [chronos_input(history, current)],
                prediction_length=1,
                quantile_levels=[0.1, 0.5, 0.9],
            )
            lower, median, upper = map(float, quantiles[0][0, 0])
            del fitted
            if not np.isfinite([lower, median, upper]).all():
                raise ValueError(f"Non-finite forecast at {current['date_us']}")

            signal = int(np.sign(median))
            actual = float(current[TARGET])
            filled = signal != 0 and pd.notna(current["open_0930"])
            records.append({
                "date_us": current["date_us"],
                "forecast_cutoff_utc": current["forecast_cutoff_utc"],
                "train_n": len(history),
                "train_through": history["date_us"].iloc[-1],
                **{name: current[name] for name in COVARIATES},
                "forecast_q10": lower,
                "forecast_log_return": median,
                "forecast_q90": upper,
                "actual_log_return": actual,
                "signal": signal,
                "entry_ts_utc": current["open_0930_ts_utc"] if filled else pd.NaT,
                "entry_price": float(current["open_0930"]) if filled else np.nan,
            })
            progress.set_postfix(status, phase="complete", q50=f"{median:+.6f}", refresh=False)
            progress.update()

    return pd.DataFrame.from_records(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-train", type=int, default=40)
    parser.add_argument("--steps", type=int, default=50, help="LoRA steps per forecast")
    parser.add_argument("--max-forecasts", type=int, help="Limit the run for a smoke check")
    parser.add_argument("--device", help="Torch device such as cuda or cpu")
    parser.add_argument("--output", type=Path, default=(
        PROJECT_ROOT / "reports" / "run_id=chronos-2" / "forecasts.parquet"
    ))
    args = parser.parse_args()
    if args.min_train < 9 or args.steps < 1 or (
        args.max_forecasts is not None and args.max_forecasts < 1
    ):
        parser.error("Require --min-train >= 9, --steps >= 1 and --max-forecasts >= 1")

    try:
        import peft  # noqa: F401  # Prevent Chronos from falling back to full tuning.
        import torch
        from chronos import Chronos2Pipeline
    except ImportError as exc:
        parser.exit(1, f"Install Chronos-2 and PEFT: pip install \"chronos-forecasting>=2.2,<3\" peft\n{exc}\n")

    overnight = build_overnight_dataset()
    daily = pd.read_parquet(PROJECT_ROOT / "data/features/daily/part-00000.parquet")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    base = load_base_model(Chronos2Pipeline, BASE_MODEL_DIR, device)
    forecasts = walk_forward(
        overnight, daily, base, min_train=args.min_train, steps=args.steps,
        max_forecasts=args.max_forecasts,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    forecasts.to_parquet(args.output, index=False)
    print(f"Wrote {len(forecasts)} forecasts to {args.output}")


if __name__ == "__main__":
    main()
