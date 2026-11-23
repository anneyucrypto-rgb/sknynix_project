"""Walk-forward Chronos-2 forecasts for the daily SKHY close-to-open return."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4
import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config import PROJECT_ROOT, PathSettings, feature_output_path
from src.features.chronos_2 import CHRONOS_FEATURE_PATH


TARGET = "overnight_return"
COVARIATES = (
    "kr_fx_adjusted_return",
    "vix_daily_return",
    "kr_proxy_flkr_return",
    "kr_proxy_usdkrw_return",
)
BASE_MODEL_ID = "amazon/chronos-2"
BASE_MODEL_DIR = PROJECT_ROOT / ".model_cache" / "chronos-2"
FINETUNED_CACHE = PROJECT_ROOT / ".model_cache" / "chronos-2-finetuned"


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
    base_pipeline,
    *,
    min_train: int,
    steps: int,
    checkpoint_cache: Path,
    max_forecasts: int | None = None,
) -> tuple[pd.DataFrame, Path]:
    from peft import get_peft_model_state_dict, set_peft_model_state_dict
    from transformers import TrainerCallback

    required = {"date_us", TARGET, *COVARIATES}
    missing = required.difference(overnight)
    if missing:
        raise ValueError(f"Overnight feature Parquet is missing columns {sorted(missing)}")
    panel = overnight.copy()
    panel["date_us"] = pd.to_datetime(panel["date_us"]).dt.date
    if panel["date_us"].isna().any() or panel["date_us"].duplicated().any():
        raise ValueError("Expected one observation per US session")
    numeric = [TARGET, *COVARIATES]
    panel[numeric] = panel[numeric].apply(pd.to_numeric, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    panel = panel.dropna(subset=list(COVARIATES)).sort_values("date_us").reset_index(drop=True)

    prior_observations = panel[TARGET].notna().cumsum().shift(fill_value=0)
    forecast_indices = panel.index[prior_observations.ge(min_train)][:max_forecasts]
    if forecast_indices.empty:
        raise ValueError("No forecast dates with enough complete prior observations")

    adapter_state = None

    class RestoreAdapter(TrainerCallback):
        def on_train_begin(self, args, state, control, model=None, **kwargs):
            # Chronos creates a new PEFT wrapper on each fit. Restore the
            # preceding origin's adapter before the first optimizer step.
            if adapter_state is not None:
                set_peft_model_state_dict(model, adapter_state)

    def fit_adapter(history: pd.DataFrame, work_dir: Path):
        nonlocal adapter_state
        fitted = base_pipeline.fit(
            inputs=[chronos_input(history)],
            prediction_length=1,
            finetune_mode="lora",
            learning_rate=1e-5,
            num_steps=steps,
            batch_size=16,
            min_past=8,
            output_dir=work_dir,
            callbacks=[RestoreAdapter()],
            save_strategy="no",
            report_to=[],
            seed=42,
            disable_tqdm=True,
            remove_printer_callback=True,
        )
        adapter_state = {
            name: weight.detach().cpu().clone()
            for name, weight in get_peft_model_state_dict(fitted.model).items()
        }
        return fitted

    records = []
    checkpoint_cache.mkdir(parents=True, exist_ok=True)
    with (
        TemporaryDirectory(prefix="staging-", dir=checkpoint_cache) as staging,
        tqdm(total=len(forecast_indices), desc="Chronos-2 test forecasts", unit="forecast") as progress,
    ):
        work_dir = Path(staging)
        for index in forecast_indices:
            current = panel.iloc[index].copy()
            history = panel.iloc[:index].dropna(subset=[TARGET]).copy()
            status = {"date": str(current["date_us"]), "train_n": len(history)}
            progress.set_postfix(status, phase="training")

            fitted = fit_adapter(history, work_dir)
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

            actual = float(current[TARGET]) if pd.notna(current[TARGET]) else np.nan
            records.append({
                "date_us": current["date_us"],
                "train_n": len(history),
                "train_through": history["date_us"].iloc[-1],
                **{name: current[name] for name in COVARIATES},
                "forecast_q10": lower,
                "forecast_log_return": median,
                "forecast_return": median,
                "forecast_q90": upper,
                "actual_log_return": actual,
                "actual_return": actual,
                "signal": int(np.sign(median)),
            })
            progress.set_postfix(status, phase="complete", q50=f"{median:+.6f}", refresh=False)
            progress.update()

        # Incorporate the last realized return only after its forecast is fixed.
        if pd.notna(current[TARGET]):
            history = panel.iloc[:index + 1].dropna(subset=[TARGET])
            progress.set_postfix(date=str(current["date_us"]), train_n=len(history), phase="final fit")
            fit_adapter(history, work_dir)

        train_through = history["date_us"].iloc[-1].isoformat()
        saved = work_dir / "finetuned-ckpt"
        (saved / "training_metadata.json").write_text(
            json.dumps({
                "base_model": BASE_MODEL_ID,
                "target": TARGET,
                "covariates": COVARIATES,
                "train_n": len(history),
                "train_through": train_through,
            }, indent=2) + "\n",
            encoding="utf-8",
        )
        checkpoint = checkpoint_cache / f"{train_through}-{uuid4().hex[:12]}"
        saved.rename(checkpoint)

    return pd.DataFrame.from_records(records), checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-train", type=int, default=40)
    parser.add_argument("--steps", type=int, default=50, help="LoRA steps per forecast")
    parser.add_argument("--max-forecasts", type=int, help="Limit the run for a smoke check")
    parser.add_argument("--device", help="Torch device such as cuda or cpu")
    parser.add_argument("--features", type=Path, default=CHRONOS_FEATURE_PATH,
                        help="Standalone overnight feature Parquet")
    parser.add_argument("--checkpoint-cache", type=Path, default=FINETUNED_CACHE,
                        help="Directory for the final LoRA checkpoints")
    parser.add_argument("--output", type=Path, default=(
        PathSettings().feature_run("chronos-2") / "forecasts.parquet"
    ))
    args = parser.parse_args()
    if args.min_train < 9 or args.steps < 1 or (
        args.max_forecasts is not None and args.max_forecasts < 1
    ):
        parser.error("Require --min-train >= 9, --steps >= 1 and --max-forecasts >= 1")
    output = feature_output_path(args.output)

    try:
        import peft  # noqa: F401  # Prevent Chronos from falling back to full tuning.
        import torch
        from chronos import Chronos2Pipeline
    except ImportError as exc:
        parser.exit(1, f"Install Chronos-2 and PEFT: pip install \"chronos-forecasting>=2.2,<3\" peft\n{exc}\n")

    if not args.features.is_file():
        parser.error(f"Feature Parquet missing at {args.features}. Run python -m src.features.chronos_2.")
    overnight = pd.read_parquet(args.features)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    base = load_base_model(Chronos2Pipeline, BASE_MODEL_DIR, device)
    forecasts, checkpoint = walk_forward(
        overnight, base, min_train=args.min_train, steps=args.steps,
        checkpoint_cache=args.checkpoint_cache, max_forecasts=args.max_forecasts,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    forecasts.to_parquet(output, index=False)
    print(f"Wrote {len(forecasts)} forecasts to {output}")
    print(f"Saved the fine-tuned LoRA checkpoint to {checkpoint}")


if __name__ == "__main__":
    main()
