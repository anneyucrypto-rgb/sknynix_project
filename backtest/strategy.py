"""Backtest a causal 09:25 premarket entry and 09:30 MOO exit."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import PathSettings, feature_output_path


EXECUTION_COLUMNS = [
    "tradable_return_0925_0930", "entry_0925", "open_0930",
    "premarket_exit_dollar_volume", "opening_dollar_volume",
    *(f"trade_high_m{offset}" for offset in range(5)),
    *(f"trade_low_m{offset}" for offset in range(5)),
]


@dataclass(frozen=True)
class PremarketStrategySpec:
    calibration_observations: int = 10
    safety_buffer_bps: float = 10.0
    transaction_cost_bps_per_side: float = 1.0
    slippage_bps_per_side: float = 3.0
    impact_coefficient_bps: float = 10.0
    risk_aversion: float = 0.0025
    max_weight: float = 1.0
    minimum_volatility_bps: float = 10.0
    stop_loss_sigma: float = 0.75
    take_profit_sigma: float = 1.50
    stop_loss_floor_bps: float = 25.0
    take_profit_floor_bps: float = 50.0
    drawdown_pause_pct: float = 1.0
    pause_sessions: int = 5
    hard_drawdown_stop_pct: float = 5.0
    initial_fund: float = 10_000.0
    minimum_fund: float = 9_000.0
    annualization_sessions: int = 252


def _calibrate_remaining_return(
    history: pd.DataFrame,
    current_forecast: float,
    minimum_volatility: float,
) -> tuple[float, float, float, float]:
    """Map the gap forecast to a 09:25–09:30 return using prior dates only."""

    x = pd.to_numeric(history["forecast_return"], errors="coerce").to_numpy(float)
    y = pd.to_numeric(
        history["tradable_return_0925_0930"], errors="coerce"
    ).to_numpy(float)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < 3:
        return math.nan, math.nan, math.nan, math.nan
    x_centered = x - x.mean()
    denominator = float(np.square(x_centered).sum())
    slope = float(np.dot(x_centered, y - y.mean()) / denominator) if denominator > 0 else 0.0
    intercept = float(y.mean() - slope * x.mean())
    residual = y - intercept - slope * x
    volatility = max(float(np.sqrt(np.mean(np.square(residual)))), minimum_volatility)
    expected = float(intercept + slope * current_forecast)
    return expected, volatility, intercept, slope


def _impact_bps(notional: float, dollar_volume: float, coefficient_bps: float) -> float:
    if notional <= 0 or not np.isfinite(dollar_volume) or dollar_volume <= 0:
        return coefficient_bps
    participation = min(notional / dollar_volume, 1.0)
    return float(coefficient_bps * math.sqrt(participation))


def _simulate_exit(
    row: pd.Series,
    position: int,
    stop_distance: float,
    take_distance: float,
) -> tuple[float, float, str]:
    """Simulate protective exits through 09:27, then lock the 09:30 MOO."""

    entry = float(row["entry_0925"])
    if position > 0:
        stop_price = entry * math.exp(-stop_distance)
        take_price = entry * math.exp(take_distance)
    else:
        stop_price = entry * math.exp(stop_distance)
        take_price = entry * math.exp(-take_distance)

    # A MOO must be submitted before 09:28. Once it is locked, the strategy no
    # longer assumes a protective order can cancel the auction instruction.
    for offset in range(3):
        high = float(row[f"trade_high_m{offset}"])
        low = float(row[f"trade_low_m{offset}"])
        stop_hit = low <= stop_price if position > 0 else high >= stop_price
        take_hit = high >= take_price if position > 0 else low <= take_price
        if stop_hit:
            asset_return = -stop_distance if position > 0 else stop_distance
            return asset_return, stop_price, f"stop_loss_m{offset}"
        if take_hit:
            asset_return = take_distance if position > 0 else -take_distance
            return asset_return, take_price, f"take_profit_m{offset}"

    exit_price = float(row["open_0930"])
    return math.log(exit_price / entry), exit_price, "moo_0930"


def evaluate_premarket_to_open_strategy(
    forecasts: pd.DataFrame,
    spec: PremarketStrategySpec = PremarketStrategySpec(),
) -> tuple[pd.DataFrame, dict[str, float | int | str]]:
    """Apply cost hurdle, volatility sizing, stops, and drawdown pauses."""

    required = {"date_us", "forecast_return", "actual_return", *EXECUTION_COLUMNS}
    missing = required.difference(forecasts.columns)
    if missing:
        raise ValueError(f"Premarket strategy input is missing: {sorted(missing)}")
    if spec.initial_fund <= 0 or spec.minimum_fund < 0:
        raise ValueError("Fund values must be nonnegative and initial_fund positive")
    if not 0 < spec.max_weight <= 1:
        raise ValueError("max_weight must be in (0, 1]")

    output = forecasts.copy().sort_values("date_us").reset_index(drop=True)
    numeric = sorted(required.difference({"date_us"}))
    output[numeric] = output[numeric].apply(pd.to_numeric, errors="coerce")
    if output[numeric].isna().any().any():
        raise ValueError("Premarket strategy input contains missing numeric values")

    fund = float(spec.initial_fund)
    peak_fund = fund
    pause_remaining = 0
    pause_events = 0
    halted = False
    rows: list[dict[str, object]] = []
    minimum_volatility = spec.minimum_volatility_bps / 10_000.0

    for index, row in output.iterrows():
        expected = volatility = calibration_intercept = calibration_slope = math.nan
        raw_weight = weight = 0.0
        position = 0
        impact_entry_bps = impact_exit_bps = execution_cost_rate = hurdle = 0.0
        stop_distance = take_distance = math.nan
        exit_price = float(row["entry_0925"])
        asset_return = gross_return = net_return = 0.0
        exit_reason = "no_trade"

        if index < spec.calibration_observations:
            status = "calibration"
        elif halted:
            status = "hard_risk_stop"
        elif pause_remaining > 0:
            status = "drawdown_pause"
            pause_remaining -= 1
        else:
            expected, volatility, calibration_intercept, calibration_slope = (
                _calibrate_remaining_return(
                    output.iloc[:index], float(row["forecast_return"]), minimum_volatility
                )
            )
            raw_weight = float(
                np.clip(
                    spec.risk_aversion * expected / (volatility ** 2),
                    -spec.max_weight,
                    spec.max_weight,
                )
            )
            candidate_notional = abs(raw_weight) * fund
            impact_entry_bps = _impact_bps(
                candidate_notional,
                float(row["premarket_exit_dollar_volume"]),
                spec.impact_coefficient_bps,
            )
            impact_exit_bps = _impact_bps(
                candidate_notional,
                float(row["opening_dollar_volume"]),
                spec.impact_coefficient_bps,
            )
            execution_cost_rate = (
                2.0 * (
                    spec.transaction_cost_bps_per_side
                    + spec.slippage_bps_per_side
                )
                + impact_entry_bps + impact_exit_bps
            ) / 10_000.0
            hurdle = execution_cost_rate + spec.safety_buffer_bps / 10_000.0
            if abs(expected) <= hurdle or raw_weight == 0:
                status = "below_cost_hurdle"
            else:
                weight = raw_weight
                position = 1 if weight > 0 else -1
                stop_distance = max(
                    spec.stop_loss_floor_bps / 10_000.0,
                    spec.stop_loss_sigma * volatility,
                )
                take_distance = max(
                    spec.take_profit_floor_bps / 10_000.0,
                    spec.take_profit_sigma * volatility,
                )
                asset_return, exit_price, exit_reason = _simulate_exit(
                    row, position, stop_distance, take_distance
                )
                gross_return = weight * asset_return
                net_return = gross_return - abs(weight) * execution_cost_rate
                status = "active"

        fund_before = fund
        fund *= math.exp(net_return)
        peak_fund = max(peak_fund, fund)
        drawdown = fund / peak_fund - 1.0
        if fund <= spec.minimum_fund or drawdown <= -spec.hard_drawdown_stop_pct / 100.0:
            halted = True
            status += ";hard_risk_stop"
        elif position and drawdown <= -spec.drawdown_pause_pct / 100.0:
            pause_remaining = spec.pause_sessions
            pause_events += 1
            status += ";pause_trigger"

        rows.append({
            **row.to_dict(),
            "calibrated_expected_return": expected,
            "predicted_volatility": volatility,
            "calibration_intercept": calibration_intercept,
            "calibration_slope": calibration_slope,
            "cost_hurdle": hurdle,
            "raw_target_weight": raw_weight,
            "target_weight": weight,
            "position": position,
            "strategy_status": status,
            "stop_loss_distance": stop_distance,
            "take_profit_distance": take_distance,
            "exit_price": exit_price,
            "exit_reason": exit_reason,
            "asset_return_0925_exit": asset_return,
            "impact_entry_bps": impact_entry_bps,
            "impact_exit_bps": impact_exit_bps,
            "execution_cost_rate": execution_cost_rate,
            "gross_strategy_return": gross_return,
            "net_strategy_return": net_return,
            "correct_traded_direction": position != 0 and gross_return > 0,
            "winning_trade": position != 0 and net_return > 0,
            "fund_before": fund_before,
            "fund_after": fund,
            "fund_drawdown": drawdown,
        })

    results = pd.DataFrame(rows)
    active = results["position"].ne(0)
    net = results["net_strategy_return"].astype(float)
    active_net = results.loc[active, "net_strategy_return"].astype(float)
    wins = active_net.loc[active_net.gt(0)]
    losses = -active_net.loc[active_net.lt(0)]
    volatility = float(net.std(ddof=1))
    downside = float(np.sqrt(np.mean(np.square(np.minimum(net, 0.0)))))
    years = len(results) / spec.annualization_sessions
    total_return = fund / spec.initial_fund - 1.0
    cagr = (fund / spec.initial_fund) ** (1.0 / years) - 1.0 if years > 0 else math.nan
    max_drawdown = float(results["fund_drawdown"].min())
    average_win = float(wins.mean()) if len(wins) else math.nan
    average_loss = float(losses.mean()) if len(losses) else math.nan
    configured_rr = results.loc[active, "take_profit_distance"].div(
        results.loc[active, "stop_loss_distance"]
    )

    metrics: dict[str, float | int | str] = {
        "strategy_mode": "premarket_to_open",
        "entry_time": "09:25 ET",
        "exit_order": "MOO submitted before 09:28 ET",
        "exit_time": "09:30 ET",
        "calibration_observations": spec.calibration_observations,
        "strategy_observations": len(results),
        "active_trades": int(active.sum()),
        "long_trades": int(results["position"].eq(1).sum()),
        "short_trades": int(results["position"].eq(-1).sum()),
        "flat_observations": int(results["position"].eq(0).sum()),
        "stop_loss_hits": int(results["exit_reason"].str.startswith("stop_loss").sum()),
        "take_profit_hits": int(results["exit_reason"].str.startswith("take_profit").sum()),
        "moo_exits": int(results["exit_reason"].eq("moo_0930").sum()),
        "direction_accuracy": float(
            results.loc[active, "correct_traded_direction"].mean()
        ) if active.any() else math.nan,
        "win_rate": float(results.loc[active, "winning_trade"].mean()) if active.any() else math.nan,
        "average_target_weight": float(results.loc[active, "target_weight"].abs().mean()) if active.any() else 0.0,
        "average_execution_cost_bps": float(
            10_000 * results.loc[active, "execution_cost_rate"].mean()
        ) if active.any() else 0.0,
        "average_configured_risk_reward": float(configured_rr.mean()) if len(configured_rr) else math.nan,
        "realized_risk_reward": average_win / average_loss if average_loss and average_loss > 0 else math.nan,
        "profit_factor": float(wins.sum() / losses.sum()) if losses.sum() > 0 else math.nan,
        "total_return": total_return,
        "cagr": float(cagr),
        "annualized_volatility": float(math.sqrt(spec.annualization_sessions) * volatility),
        "sharpe_ratio": float(math.sqrt(spec.annualization_sessions) * net.mean() / volatility) if volatility > 0 else math.nan,
        "sortino_ratio": float(math.sqrt(spec.annualization_sessions) * net.mean() / downside) if downside > 0 else math.nan,
        "calmar_ratio": float(cagr / abs(max_drawdown)) if max_drawdown < 0 else math.nan,
        "max_drawdown": max_drawdown,
        "pause_events": pause_events,
        "paused_sessions": int(results["strategy_status"].eq("drawdown_pause").sum()),
        "hard_stop_triggered": int(halted),
        "initial_fund": spec.initial_fund,
        "ending_fund": fund,
        "minimum_fund": spec.minimum_fund,
        "overnight_forecast_direction_accuracy": float(
            np.sign(results["forecast_return"]).eq(np.sign(results["actual_return"])).mean()
        ),
    }
    return results, metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="overnight-best")
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--input", type=Path, help="Single forecast Parquet file")
    inputs.add_argument(
        "--forecast", nargs=2, action="append", metavar=("MODEL", "PARQUET"),
        help="Named forecast file containing 09:25 close-to-open log forecasts. Repeat for each model.",
    )
    parser.add_argument(
        "--execution-data", type=Path,
        help="Optional overnight execution panel. Defaults to rebuilding from raw bars when needed.",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--calibration-observations", type=int, default=10)
    parser.add_argument("--safety-buffer-bps", type=float, default=10.0)
    parser.add_argument("--transaction-cost-bps-per-side", type=float, default=1.0)
    parser.add_argument("--slippage-bps-per-side", type=float, default=3.0)
    parser.add_argument("--impact-coefficient-bps", type=float, default=10.0)
    parser.add_argument("--risk-aversion", type=float, default=0.0025)
    parser.add_argument("--max-weight", type=float, default=1.0)
    parser.add_argument("--stop-loss-sigma", type=float, default=0.75)
    parser.add_argument("--take-profit-sigma", type=float, default=1.50)
    parser.add_argument("--drawdown-pause-pct", type=float, default=1.0)
    parser.add_argument("--pause-sessions", type=int, default=5)
    parser.add_argument("--hard-drawdown-stop-pct", type=float, default=5.0)
    parser.add_argument("--initial-fund", type=float, default=10_000.0)
    parser.add_argument("--minimum-fund", type=float, default=9_000.0)
    return parser.parse_args()


def _session_index(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    dates = pd.to_datetime(frame["date_us"], errors="raise").dt.normalize()
    if dates.isna().any() or dates.duplicated().any():
        raise ValueError(f"{source} must contain one valid row per US session")
    return frame.assign(date_us=dates.dt.date)


def _load_execution_data(path: Path | None) -> pd.DataFrame:
    if path is None:
        from src.features.utils import build_overnight_features

        execution = build_overnight_features()
    else:
        execution = pd.read_parquet(path)
    if "actual_return" not in execution:
        execution = execution.rename(columns={"overnight_return": "actual_return"})
    required = ["date_us", "actual_return", *EXECUTION_COLUMNS]
    return _session_index(execution[required], str(path or "Raw execution panel"))


def evaluate_forecast_files(
    sources: list[tuple[str, Path]],
    spec: PremarketStrategySpec,
    execution_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate each file with its own calibration history and portfolio."""
    execution = None
    results_by_model = []
    summaries = []
    for model, path in sources:
        forecasts = _session_index(pd.read_parquet(path), str(path))
        if "forecast_return" not in forecasts:
            forecasts = forecasts.rename(columns={"forecast_log_return": "forecast_return"})
        if "forecast_return" not in forecasts:
            raise ValueError(f"{path} needs forecast_return or forecast_log_return")
        input_count = len(forecasts)
        if execution_path is not None or not {"actual_return", *EXECUTION_COLUMNS}.issubset(forecasts):
            if execution is None:
                execution = _load_execution_data(execution_path)
            # Derive realized returns and fills from the execution panel.
            # Forecast files can carry their own simulated entry fields.
            forecasts = forecasts.drop(
                columns=["actual_return", *EXECUTION_COLUMNS], errors="ignore"
            ).merge(execution, on="date_us", how="left", validate="one_to_one")
        numeric = ["forecast_return", "actual_return", *EXECUTION_COLUMNS]
        forecasts[numeric] = forecasts[numeric].apply(pd.to_numeric, errors="coerce")
        complete = np.isfinite(forecasts[numeric].to_numpy(dtype=float, na_value=np.nan)).all(axis=1)
        forecasts = forecasts.loc[complete]
        if forecasts.empty:
            raise ValueError(f"{model} has no forecast sessions with complete execution data")

        results, metrics = evaluate_premarket_to_open_strategy(forecasts, spec)
        results_by_model.append(results.assign(forecast_model=model))
        summaries.append({
            "forecast_model": model,
            "forecast_file": str(path),
            "input_forecasts": input_count,
            "excluded_sessions": input_count - len(results),
            "first_session": results["date_us"].iloc[0],
            "last_session": results["date_us"].iloc[-1],
            **metrics,
        })
    return pd.concat(results_by_model, ignore_index=True), pd.DataFrame(summaries)


def _write_trading_results(
    data_dir: Path,
    output: Path,
    run_id: str,
    metrics: dict[str, float | int | str],
    spec: PremarketStrategySpec,
) -> Path:
    """Write the retained model's executable-strategy report."""

    backtest = pd.read_parquet(data_dir / "overnight_backtest_summary.parquet").iloc[0]
    h2 = pd.read_parquet(data_dir / "h2_model.parquet").iloc[0]

    def number(name: str, digits: int = 4) -> str:
        value = float(metrics[name])
        return "—" if not np.isfinite(value) else f"{value:.{digits}f}"

    active_trades = int(metrics["active_trades"])
    profitable = (
        active_trades >= 3
        and float(metrics["total_return"]) > 0
        and float(metrics["sharpe_ratio"]) > 0
    )
    conclusion = (
        "The strategy passes the minimum economic screen, but still requires continued forward validation."
        if profitable
        else "The strategy does not pass the minimum economic screen of at least three trades, positive net return, and positive Sharpe."
    )
    lines = [
        "# Premarket-to-open trading results",
        "",
        f"**Run ID:** `{run_id}`  ",
        f"**Generated:** {datetime.now(timezone.utc).isoformat()}  ",
        "**Forecast model:** retained bivariate-H1/SOXX-gauge H2  ",
        "**Trading interval:** 09:25 premarket entry to 09:30 opening auction",
        "",
        "## Forecast input",
        "",
        "The strategy uses H2 forecasts. H1 supplies only the equilibrium benchmark used inside H2.",
        "",
        "| Forecast measure | Result |",
        "| --- | ---: |",
        f"| H2 specification | `{h2['h2_spec']}` |",
        f"| Complete observations | {int(h2['common_comparison_n'])} |",
        f"| Expanding-window forecasts | {int(backtest['n_forecasts'])} |",
        f"| RMSE | {float(backtest['rmse']):.6f} |",
        f"| MAE | {float(backtest['mae']):.6f} |",
        f"| Forecast directional accuracy | {100 * float(backtest['direction_accuracy']):.1f}% |",
        f"| OOS R² versus zero | {float(backtest['oos_r2_zero']):.4f} |",
        f"| H2 adjustment coefficient | {float(h2['ect_coefficient']):.6f} |",
        f"| H2 adjustment p-value | {float(h2['ect_p']):.2e} |",
        "",
        "## Execution rules",
        "",
        "1. Freeze the H2 forecast at 09:25 ET and calibrate it to the remaining 09:25–09:30 return using prior observations only.",
        "2. Enter at the 09:25 SKHY premarket open only when expected profit exceeds transaction costs, slippage, impact, and the safety buffer.",
        "3. Keep volatility-scaled protective stop-loss and take-profit orders active through 09:27.",
        "4. Submit an opposite-side MOO before 09:28 for every survivor and exit at 09:30.",
        "5. Apply volatility-scaled sizing, drawdown pauses, the hard drawdown stop, and the minimum-fund floor.",
        "",
        "Only the executable 09:25–09:30 return is credited. The already-realized portion of the full overnight gap is excluded.",
        "",
        "## Cost and risk assumptions",
        "",
        "| Assumption | Value |",
        "| --- | ---: |",
        f"| Safety buffer | {spec.safety_buffer_bps:.2f} bps |",
        f"| Transaction cost | {spec.transaction_cost_bps_per_side:.2f} bps per side |",
        f"| Slippage | {spec.slippage_bps_per_side:.2f} bps per side |",
        f"| Impact coefficient | {spec.impact_coefficient_bps:.2f} bps |",
        f"| Maximum weight | {100 * spec.max_weight:.1f}% |",
        f"| Stop / take-profit volatility multipliers | {spec.stop_loss_sigma:.2f} / {spec.take_profit_sigma:.2f} |",
        f"| Drawdown pause threshold | {spec.drawdown_pause_pct:.2f}% |",
        f"| Hard drawdown stop | {spec.hard_drawdown_stop_pct:.2f}% |",
        "",
        "## Trading performance",
        "",
        "| Measure | Result |",
        "| --- | ---: |",
        f"| Forecast observations | {int(metrics['strategy_observations'])} |",
        f"| Active trades | {active_trades} |",
        f"| Long / short / flat | {int(metrics['long_trades'])} / {int(metrics['short_trades'])} / {int(metrics['flat_observations'])} |",
        f"| Stop / take-profit / MOO exits | {int(metrics['stop_loss_hits'])} / {int(metrics['take_profit_hits'])} / {int(metrics['moo_exits'])} |",
        f"| Traded directional accuracy | {100 * float(metrics['direction_accuracy']):.1f}% |",
        f"| Win rate | {100 * float(metrics['win_rate']):.1f}% |",
        f"| Net cumulative return | {100 * float(metrics['total_return']):.2f}% |",
        f"| CAGR | {100 * float(metrics['cagr']):.2f}% |",
        f"| Sharpe ratio | {number('sharpe_ratio')} |",
        f"| Sortino ratio | {number('sortino_ratio')} |",
        f"| Calmar ratio | {number('calmar_ratio')} |",
        f"| Maximum drawdown | {100 * float(metrics['max_drawdown']):.2f}% |",
        f"| Realized risk-to-reward | {number('realized_risk_reward', 3)} |",
        f"| Profit factor | {number('profit_factor', 3)} |",
        f"| Average execution cost | {float(metrics['average_execution_cost_bps']):.2f} bps |",
        f"| Initial fund | ${float(metrics['initial_fund']):,.2f} |",
        f"| Remaining fund | ${float(metrics['ending_fund']):,.2f} |",
        "",
        f"**Conclusion:** {conclusion}",
        "",
        "Nasdaq's MOO timing is documented in the [Opening/Closing Cross FAQ](https://beta.nasdaqtrader.com/content/ProductsServices/Trading/Crosses/openclose_faqs.pdf). The one-minute 09:30 open remains an auction-price proxy rather than an order-level fill record.",
        "",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def _write_forecast_comparison(summary: pd.DataFrame, output: Path) -> None:
    lines = [
        "# Premarket-to-open forecast strategy results",
        "",
        "Each forecast file was evaluated independently under the 09:25–09:30 ET rules.",
        "",
        "| Model | Evaluated sessions | Excluded | Trades | Net return | Sharpe | Ending fund |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary.itertuples(index=False):
        model = row.forecast_model.replace("|", r"\|").replace("\n", " ")
        sharpe = f"{row.sharpe_ratio:.2f}" if np.isfinite(row.sharpe_ratio) else "—"
        lines.append(
            f"| {model} | {row.first_session}–{row.last_session} "
            f"({row.strategy_observations}) | {row.excluded_sessions} | {row.active_trades} "
            f"| {row.total_return:.2%} | {sharpe} | ${row.ending_fund:,.2f} |"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    data_dir = PathSettings(project_root).feature_run(args.run_id)
    reports = project_root / "reports"
    default_run = args.input is None and args.forecast is None and args.execution_data is None
    if args.forecast:
        sources = [(name, Path(path)) for name, path in args.forecast]
    else:
        input_path = args.input or data_dir / "overnight_backtest.parquet"
        sources = [(input_path.stem if args.input else "h2", input_path)]
    names = [name for name, _ in sources]
    if len(names) != len(set(names)):
        raise SystemExit("Use a unique name for each --forecast")
    output_name = "premarket_to_open_strategy.parquet" if default_run else "forecast_strategies.parquet"
    output_path = feature_output_path(args.output or data_dir / output_name, project_root)
    summary_path = output_path.with_name(output_path.stem + "_summary.parquet")
    input_paths = {path.resolve() for _, path in sources}
    if args.execution_data is not None:
        input_paths.add(args.execution_data.resolve())
    if input_paths.intersection({output_path.resolve(), summary_path.resolve()}):
        raise SystemExit("Choose an output path that differs from the input files")
    spec = PremarketStrategySpec(
        calibration_observations=args.calibration_observations,
        safety_buffer_bps=args.safety_buffer_bps,
        transaction_cost_bps_per_side=args.transaction_cost_bps_per_side,
        slippage_bps_per_side=args.slippage_bps_per_side,
        impact_coefficient_bps=args.impact_coefficient_bps,
        risk_aversion=args.risk_aversion,
        max_weight=args.max_weight,
        stop_loss_sigma=args.stop_loss_sigma,
        take_profit_sigma=args.take_profit_sigma,
        drawdown_pause_pct=args.drawdown_pause_pct,
        pause_sessions=args.pause_sessions,
        hard_drawdown_stop_pct=args.hard_drawdown_stop_pct,
        initial_fund=args.initial_fund,
        minimum_fund=args.minimum_fund,
    )
    results, summary = evaluate_forecast_files(sources, spec, args.execution_data)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_parquet(output_path, index=False)
    summary.to_parquet(summary_path, index=False)
    print(summary.set_index("forecast_model").T.to_string(), flush=True)
    print(f"[STRATEGY] Wrote {output_path}", flush=True)
    print(f"[STRATEGY] Wrote {summary_path}", flush=True)
    if default_run:
        trading_report = _write_trading_results(
            data_dir, reports / "TRADING_RESULTS.md", args.run_id, summary.iloc[0].to_dict(), spec
        )
    else:
        trading_report = reports / (output_path.stem + "_summary.md")
        _write_forecast_comparison(summary, trading_report)
    print(f"[STRATEGY] Wrote {trading_report}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
