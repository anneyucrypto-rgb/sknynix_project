"""Estimate event-time ARDL Granger models and control-group ablations."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Final, Iterable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import statsmodels.api as sm


PROJECT_DIR: Final = Path(__file__).resolve().parents[1]
GRANGER_PANEL_PATH: Final = PROJECT_DIR / "data" / "processed" / "granger_panel.parquet"
RESULTS_DIR: Final = PROJECT_DIR / "models" / "results"
MODEL_DOC_PATH: Final = PROJECT_DIR / "design" / "model.md"
SUMMARY_DOC_PATH: Final = PROJECT_DIR / "design" / "model_summary.md"
HORIZONS: Final = (5, 10, 15, 30, 45, 60, 100, 120)
ANCHORS: Final = ("premarket_dynamic", "regular_0930")
LAG_SELECTION_CRITERION: Final = "bic"
BASE_CONTROLS: Final = ("skhy_pre_event_return",)
MACRO_COMMON: Final = (
    "kospi_return_latest",
    "usdkrw_return_latest",
    "vix_level_lag1d",
    "vix_change_lag1d",
    "us_2y_yield_change_lag1d",
    "us_10y_yield_change_lag1d",
)
KOREAN_MACRO_CONTROLS: Final = (
    "kospi_return_latest",
    "usdkrw_return_latest",
)
LIQUIDITY_CONTROLS: Final = (
    "skhy_pre_event_volume_z",
    "skhy_vwap_deviation",
)
MICROSTRUCTURE_CONTROLS: Final = (
    "skhy_pre_event_rv_30m",
    *LIQUIDITY_CONTROLS,
)


def control_groups(anchor: str) -> dict[str, tuple[str, ...]]:
    """Return anchor-specific, strictly pre-event control blocks."""
    spy_return = "spy_pre_event_return_5m"
    sector_return = "soxx_pre_event_return_5m"
    return {
        "macro_only": (spy_return, *MACRO_COMMON),
        "korean_macro_only": KOREAN_MACRO_CONTROLS,
        "sector_only": (sector_return,),
        "liquidity_only": LIQUIDITY_CONTROLS,
        "microstructure_only": MICROSTRUCTURE_CONTROLS,
    }


ALL_GROUP_COLUMNS: Final = {
    column
    for anchor in ANCHORS
    for columns in control_groups(anchor).values()
    for column in columns
}


def load_granger_panel(path: Path = GRANGER_PANEL_PATH) -> pd.DataFrame:
    """Load and validate the event-level feature panel."""
    frame = pd.read_parquet(path).copy()
    required = {
        "event_date_ny",
        "anchor_type",
        "kr_return_0",
        "kr_return_lag1",
        "kr_return_lag2",
        *BASE_CONTROLS,
        *ALL_GROUP_COLUMNS,
        *(f"skhy_fwd_return_{horizon}m" for horizon in HORIZONS),
        *(
            f"skhy_fwd_return_{horizon}m_lag{lag}"
            for horizon in HORIZONS
            for lag in (1, 2)
        ),
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Granger panel is missing columns: {sorted(missing)}")
    frame["event_date_ny"] = pd.to_datetime(frame["event_date_ny"])
    frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True)
    return frame.sort_values(["anchor_type", "event_date_ny"]).reset_index(drop=True)


def _fit_ols(
    sample: pd.DataFrame,
    response: str,
    predictors: list[str],
    hac_maxlags: int,
) -> tuple[object, object]:
    design = sm.add_constant(sample[predictors].astype(float), has_constant="add")
    fitted = sm.OLS(sample[response].astype(float), design).fit()
    robust = fitted.get_robustcov_results(cov_type="HAC", maxlags=hac_maxlags)
    return fitted, robust


def _parameter(robust: object, names: Iterable[str], term: str) -> tuple[float, float, float]:
    names = list(names)
    position = names.index(term)
    return (
        float(robust.params[position]),
        float(robust.bse[position]),
        float(robust.pvalues[position]),
    )


def _joint_korean_test(
    robust: object, parameter_names: list[str], korean_columns: list[str]
) -> tuple[float, float]:
    restriction = np.zeros((len(korean_columns), len(parameter_names)))
    for row, column in enumerate(korean_columns):
        restriction[row, parameter_names.index(column)] = 1.0
    try:
        test = robust.wald_test(restriction, use_f=True, scalar=True)
        return float(test.statistic), float(test.pvalue)
    except (ValueError, np.linalg.LinAlgError):
        return np.nan, np.nan


def _select_orders(
    sample: pd.DataFrame,
    response: str,
    ar_columns: list[str],
    max_p: int,
    max_q: int,
    hac_maxlags: int,
) -> tuple[int, int]:
    candidates: list[tuple[float, int, int]] = []
    kr_columns = ["kr_return_0", "kr_return_lag1", "kr_return_lag2"]
    for p_order in range(max_p + 1):
        for q_order in range(max_q + 1):
            predictors = [
                *ar_columns[:p_order],
                *kr_columns[: q_order + 1],
                *BASE_CONTROLS,
            ]
            if len(sample) <= len(predictors) + 5:
                continue
            fitted, _ = _fit_ols(sample, response, predictors, hac_maxlags)
            score = float(fitted.bic)
            candidates.append((score, p_order, q_order))
    if not candidates:
        raise ValueError("No admissible ARDL lag-order candidate")
    _, p_order, q_order = min(candidates, key=lambda value: value[0])
    return p_order, q_order


def _fit_specification(
    sample: pd.DataFrame,
    *,
    anchor: str,
    horizon: int,
    response: str,
    ablation: str,
    controls: tuple[str, ...],
    p_order: int,
    q_order: int,
    hac_maxlags: int,
) -> dict[str, object]:
    ar_columns = [
        f"skhy_fwd_return_{horizon}m_lag1",
        f"skhy_fwd_return_{horizon}m_lag2",
    ]
    korean_columns = ["kr_return_0", "kr_return_lag1", "kr_return_lag2"][
        : q_order + 1
    ]
    predictors = [
        *ar_columns[:p_order],
        *korean_columns,
        *BASE_CONTROLS,
        *controls,
    ]
    fitted, robust = _fit_ols(sample, response, predictors, hac_maxlags)
    parameter_names = list(fitted.params.index)
    beta, std_error, pvalue = _parameter(
        robust, parameter_names, "kr_return_0"
    )
    wald_stat, wald_pvalue = _joint_korean_test(
        robust, parameter_names, korean_columns
    )
    return {
        "anchor_type": anchor,
        "horizon_minutes": horizon,
        "response_column": response,
        "ablation": ablation,
        "controls": ", ".join(controls),
        "status": "estimated",
        "selection_criterion": LAG_SELECTION_CRITERION,
        "p_order": p_order,
        "q_order": q_order,
        "n_obs": int(fitted.nobs),
        "sample_start": sample["event_date_ny"].min().date().isoformat(),
        "sample_end": sample["event_date_ny"].max().date().isoformat(),
        "beta_kr_current": beta,
        "beta_kr_current_std_error_hac": std_error,
        "beta_kr_current_pvalue_hac": pvalue,
        "korean_block_wald_stat_hac": wald_stat,
        "korean_block_wald_pvalue_hac": wald_pvalue,
        "adjusted_r2": float(fitted.rsquared_adj),
        "aic": float(fitted.aic),
        "bic": float(fitted.bic),
    }


def run_ablations(
    panel: pd.DataFrame,
    *,
    max_p: int = 2,
    max_q: int = 2,
    hac_maxlags: int = 2,
) -> pd.DataFrame:
    """Run baseline and one-group-at-a-time ARDL comparisons."""
    records: list[dict[str, object]] = []
    kr_max = ["kr_return_0", "kr_return_lag1", "kr_return_lag2"][: max_q + 1]

    for anchor in ANCHORS:
        anchor_sample = panel[panel["anchor_type"] == anchor].copy()
        for horizon in HORIZONS:
            response = f"skhy_fwd_return_{horizon}m"
            ar_max = [
                f"skhy_fwd_return_{horizon}m_lag{lag}"
                for lag in range(1, max_p + 1)
            ]
            available_groups: dict[str, tuple[str, ...]] = {}
            unavailable_groups: dict[str, tuple[str, ...]] = {}
            for name, columns in control_groups(anchor).items():
                if all(anchor_sample[column].notna().any() for column in columns):
                    available_groups[name] = columns
                else:
                    unavailable_groups[name] = columns

            common_columns = [
                response,
                *ar_max,
                *kr_max,
                *BASE_CONTROLS,
                *(
                    column
                    for columns in available_groups.values()
                    for column in columns
                ),
            ]
            common_sample = anchor_sample.dropna(subset=common_columns).copy()
            if common_sample.empty:
                for name, controls in {
                    "baseline": (),
                    **control_groups(anchor),
                }.items():
                    records.append(
                        {
                            "anchor_type": anchor,
                            "horizon_minutes": horizon,
                            "response_column": response,
                            "ablation": name,
                            "controls": ", ".join(controls),
                            "status": "unavailable_anchor_or_sample",
                            "selection_criterion": LAG_SELECTION_CRITERION,
                            "p_order": np.nan,
                            "q_order": np.nan,
                            "n_obs": 0,
                            "sample_start": None,
                            "sample_end": None,
                            "beta_kr_current": np.nan,
                            "beta_kr_current_std_error_hac": np.nan,
                            "beta_kr_current_pvalue_hac": np.nan,
                            "korean_block_wald_stat_hac": np.nan,
                            "korean_block_wald_pvalue_hac": np.nan,
                            "adjusted_r2": np.nan,
                            "aic": np.nan,
                            "bic": np.nan,
                            "beta_change_from_baseline": np.nan,
                            "abs_beta_change_from_baseline": np.nan,
                            "pct_beta_change_from_baseline": np.nan,
                            "largest_abs_change_in_cell": False,
                        }
                    )
                continue

            p_order, q_order = _select_orders(
                common_sample,
                response,
                ar_max,
                max_p,
                max_q,
                hac_maxlags,
            )
            cell_records = [
                _fit_specification(
                    common_sample,
                    anchor=anchor,
                    horizon=horizon,
                    response=response,
                    ablation="baseline",
                    controls=(),
                    p_order=p_order,
                    q_order=q_order,
                    hac_maxlags=hac_maxlags,
                )
            ]
            for name, controls in available_groups.items():
                cell_records.append(
                    _fit_specification(
                        common_sample,
                        anchor=anchor,
                        horizon=horizon,
                        response=response,
                        ablation=name,
                        controls=controls,
                        p_order=p_order,
                        q_order=q_order,
                        hac_maxlags=hac_maxlags,
                    )
                )

            baseline_beta = float(cell_records[0]["beta_kr_current"])
            for record in cell_records:
                change = float(record["beta_kr_current"]) - baseline_beta
                record["beta_change_from_baseline"] = change
                record["abs_beta_change_from_baseline"] = abs(change)
                record["pct_beta_change_from_baseline"] = (
                    100.0 * change / abs(baseline_beta)
                    if abs(baseline_beta) > 1e-12
                    else np.nan
                )
                record["largest_abs_change_in_cell"] = False

            nonbaseline = [
                record for record in cell_records if record["ablation"] != "baseline"
            ]
            if nonbaseline:
                largest = max(
                    nonbaseline,
                    key=lambda record: record["abs_beta_change_from_baseline"],
                )
                largest["largest_abs_change_in_cell"] = True
            records.extend(cell_records)

            for name, controls in unavailable_groups.items():
                records.append(
                    {
                        "anchor_type": anchor,
                        "horizon_minutes": horizon,
                        "response_column": response,
                        "ablation": name,
                        "controls": ", ".join(controls),
                        "status": "unavailable_controls",
                        "selection_criterion": LAG_SELECTION_CRITERION,
                        "p_order": p_order,
                        "q_order": q_order,
                        "n_obs": 0,
                        "sample_start": None,
                        "sample_end": None,
                        "beta_kr_current": np.nan,
                        "beta_kr_current_std_error_hac": np.nan,
                        "beta_kr_current_pvalue_hac": np.nan,
                        "korean_block_wald_stat_hac": np.nan,
                        "korean_block_wald_pvalue_hac": np.nan,
                        "adjusted_r2": np.nan,
                        "aic": np.nan,
                        "bic": np.nan,
                        "beta_change_from_baseline": np.nan,
                        "abs_beta_change_from_baseline": np.nan,
                        "pct_beta_change_from_baseline": np.nan,
                        "largest_abs_change_in_cell": False,
                    }
                )

    return pd.DataFrame.from_records(records).sort_values(
        ["anchor_type", "horizon_minutes", "ablation"]
    ).reset_index(drop=True)


def _markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    def clean(value: object) -> str:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return "—"
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    lines.extend("| " + " | ".join(clean(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def write_model_design(path: Path, max_p: int, max_q: int) -> None:
    text = f"""# Event-time ARDL Granger ablation design

## Objective

Test whether the latest available Korean SK hynix return predicts future U.S.
SKHY returns and identify which control group changes the contemporaneous
Korean coefficient most.

## Equation

For anchor `a` and horizon `h`:

$$
\\begin{{aligned}}
r_{{d,a}}^{{US,h}}
&= \\alpha_{{a,h}}
+ \\sum_{{i=1}}^p \\phi_{{i,a,h}} r_{{d-i,a}}^{{US,h}} \\\\
&\\quad + \\sum_{{j=0}}^q \\beta_{{j,a,h}} r_{{d-j}}^{{KR}}
+ \\rho_{{a,h}} r_{{d,a}}^{{SKHY,pre}} \\\\
&\\quad + \\boldsymbol{{\\gamma}}_{{a,h}}^\\mathsf{{T}} X_{{d,a}}
+ \\varepsilon_{{d,a,h}} .
\\end{{aligned}}
$$

The dependent variable is
`skhy_fwd_return_{{h}}m = ln(P[t0+h]/P[t0])` for

$$
H = \\left\{{5,10,15,30,45,60,100,120\\right\}}\\ \\text{{minutes}}.
$$

Models are estimated separately at `premarket_dynamic` and `regular_0930`,
yielding 16 conceptual anchor--horizon observations per U.S. trading day before
missing-price and execution-availability filters.

## Event anchors

| Anchor | Selection | Purpose |
|---|---|---|
| `premarket_dynamic` | Earliest eligible five-minute decision timestamp from 07:00 through 09:20 ET | Detect Korean-driven price discovery during the Schwab premarket session |
| `regular_0930` | Fixed 09:30 ET research timestamp | Measure residual price discovery during regular trading |

The dynamic candidate set is

$$
\\mathcal{{T}}_d=\\{{t:07\\!:\\!00\\leq t<09\\!:\\!25,\\ t\\text{{ is observable}}\\}}.
$$

At each candidate, the latest observable quote must have a positive bid, an ask
strictly above the bid, spread no greater than `S_max`, combined displayed depth
at least `D_min`, and quote age no greater than `A_max`. Completed-bar volume
over `[t-L,t)` must be positive. The selected anchor is

$$
t_d^*=\\min\\{{t\\in\\mathcal{{T}}_d:\\text{{ all liquidity filters pass}}\\}}.
$$

The default, predeclared thresholds are `S_max = 100` bps, `D_min = 100`
displayed-size units, `A_max = 60` seconds, and `L = 5` minutes. They are
configurable in `ingestion/features.py` and use no return or future-price
information. If full quote ticks are absent or no candidate qualifies, the
anchor is recorded as unavailable rather than replaced by an invented time or
price.

At 09:30, the five-minute bar open is only a research reference price. First
trade and quote fields remain separate execution diagnostics and are not
treated as guaranteed fills.

## Predictor blocks

- U.S. AR block: same-anchor, same-horizon response lags 1 and 2.
- Korean block: `kr_return_0`, `kr_return_lag1`, `kr_return_lag2`.
- Own-history control: `skhy_pre_event_return`.
- Macro-only: pre-event SPY, KOSPI return, USD/KRW return, lagged VIX level
  and change, and lagged U.S. 2Y/10Y yield changes.
- Korean-macro-only: KOSPI return and USD/KRW return.
- Sector-only: pre-event SOXX return.
- Liquidity-only: `skhy_pre_event_volume_z` and `skhy_vwap_deviation`.
- Microstructure-only: `skhy_pre_event_rv_30m` plus the two liquidity fields.

At both anchors, SPY and SOXX use exact five-minute returns ending at the
anchor. Korean 3Y/10Y yields and a Korean sector index are not
included because no validated daily source is currently ingested; they are
reported as unavailable rather than replaced with proxies.

Dynamic-anchor spread, depth, quote age, and trailing volume are eligibility
variables only. Stored post-anchor trade/quote windows are outcomes or mediators
and are excluded from predictive controls to prevent look-ahead bias.

The baseline contains the intercept, selected U.S. AR terms, Korean block, and
own-history control. Each ablation adds exactly one control group.

## Comparison protocol

For each anchor/horizon, all estimable specifications use the same complete-case
dates. Lag orders are selected once from the unrestricted baseline over
`p = 0..{max_p}` and `q = 0..{max_q}` by minimum BIC, then held
fixed across ablations. The reported change is

$$
\\Delta \\beta_0^g
= \\widehat{{\\beta}}_0^g
- \\widehat{{\\beta}}_0^{{\\mathrm{{baseline}}}} .
$$

and the group with the largest $|\\Delta\\beta_0^g|$ changes the Korean
coefficient most for that anchor/horizon. Inference uses OLS coefficients with
HAC standard errors. The Korean-block Wald test evaluates

$$
H_0:\\quad \\beta_0 = \\beta_1 = \\cdots = \\beta_q = 0 .
$$

Unavailable dynamic anchors and incomplete pre-event windows are marked
unavailable, not zero-filled. Results are exploratory because the current
sample is short and tests multiple horizons.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_model_summary(results: pd.DataFrame, path: Path) -> None:
    estimated = results[results["status"] == "estimated"].copy()
    changes = estimated[estimated["ablation"] != "baseline"].copy()
    baseline_results = estimated[estimated["ablation"] == "baseline"].copy()
    group_summary = (
        changes.groupby("ablation")["abs_beta_change_from_baseline"]
        .agg(["count", "mean", "median", "max"])
        .reset_index()
    )
    group_rows = [
        [
            row.ablation,
            int(row["count"]),
            f"{row['mean']:.4f}",
            f"{row['median']:.4f}",
            f"{row['max']:.4f}",
        ]
        for _, row in group_summary.iterrows()
    ]
    regular_summary = (
        changes[changes["anchor_type"] == "regular_0930"]
        .groupby("ablation")["abs_beta_change_from_baseline"]
        .agg(["count", "mean", "median", "max"])
        .reset_index()
    )
    regular_rows = [
        [
            row.ablation,
            int(row["count"]),
            f"{row['mean']:.4f}",
            f"{row['median']:.4f}",
            f"{row['max']:.4f}",
        ]
        for _, row in regular_summary.iterrows()
    ]
    winner_counts = (
        changes[changes["largest_abs_change_in_cell"]]
        .groupby("ablation")
        .size()
        .sort_values(ascending=False)
    )
    winner_rows = [[group, int(count)] for group, count in winner_counts.items()]
    overall_winner = str(winner_counts.index[0])

    headline_rows: list[list[object]] = []
    for (anchor, horizon), cell in estimated.groupby(
        ["anchor_type", "horizon_minutes"], sort=True
    ):
        baseline = cell[cell["ablation"] == "baseline"].iloc[0]
        group_changes = {
            row.ablation: row.abs_beta_change_from_baseline
            for _, row in cell[cell["ablation"] != "baseline"].iterrows()
        }
        largest = (
            max(group_changes, key=group_changes.get) if group_changes else "—"
        )
        headline_rows.append(
            [
                anchor,
                horizon,
                f"{baseline.beta_kr_current:.4f}",
                f"{group_changes.get('macro_only', np.nan):.4f}"
                if "macro_only" in group_changes
                else "—",
                f"{group_changes.get('korean_macro_only', np.nan):.4f}"
                if "korean_macro_only" in group_changes
                else "—",
                f"{group_changes.get('sector_only', np.nan):.4f}"
                if "sector_only" in group_changes
                else "—",
                f"{group_changes.get('liquidity_only', np.nan):.4f}"
                if "liquidity_only" in group_changes
                else "—",
                f"{group_changes.get('microstructure_only', np.nan):.4f}"
                if "microstructure_only" in group_changes
                else "—",
                largest,
                f"p={int(baseline.p_order)}, q={int(baseline.q_order)}",
                int(baseline.n_obs),
            ]
        )

    unavailable = results[results["status"] != "estimated"]
    unavailable_text = (
        ", ".join(
            f"{row.anchor_type}/{int(row.horizon_minutes)}m/{row.ablation}"
            for _, row in unavailable.iterrows()
        )
        or "None"
    )
    baseline_rows = [
        [
            row.anchor_type,
            int(row.horizon_minutes),
            f"{row.beta_kr_current:.4f}",
            f"{row.beta_kr_current_pvalue_hac:.4f}",
            f"{row.korean_block_wald_pvalue_hac:.4f}",
            f"p={int(row.p_order)}, q={int(row.q_order)}",
            int(row.n_obs),
        ]
        for _, row in baseline_results.sort_values(
            ["anchor_type", "horizon_minutes"]
        ).iterrows()
    ]

    def significance_sentence(anchor: str) -> str:
        anchor_rows = baseline_results[baseline_results["anchor_type"] == anchor]
        if anchor_rows.empty:
            return f"At `{anchor}`, no baseline model is estimable in this run."
        beta_horizons = anchor_rows.loc[
            anchor_rows["beta_kr_current_pvalue_hac"] < 0.05, "horizon_minutes"
        ].astype(int).tolist()
        block_horizons = anchor_rows.loc[
            anchor_rows["korean_block_wald_pvalue_hac"] < 0.05, "horizon_minutes"
        ].astype(int).tolist()

        def describe(values: list[int]) -> str:
            if not values:
                return "none of the"
            labels = [f"{value}-minute" for value in values]
            joined = labels[0] if len(labels) == 1 else ", ".join(labels[:-1]) + f" and {labels[-1]}"
            return f"the {joined}"

        return (
            f"At `{anchor}`, the current Korean coefficient is significant at "
            f"5% for {describe(beta_horizons)} responses; the selected Korean "
            f"block is jointly significant for {describe(block_horizons)} responses."
        )

    inference_text = "\n\n".join(
        significance_sentence(anchor) for anchor in ANCHORS
    )
    singapore_tz = ZoneInfo("Asia/Singapore")
    generated = datetime.now(singapore_tz).strftime("%Y-%m-%d %H:%M:%S SGT")
    text = f"""# Granger ablation results summary

Generated: {generated}

## Overall control-group influence

The table ranks groups using the absolute coefficient change

$$
\\left|\\Delta \\beta_0^g\\right|
= \\left|
\\widehat{{\\beta}}_0^g
- \\widehat{{\\beta}}_0^{{\\mathrm{{baseline}}}}
\\right| .
$$

A larger change is sensitivity, not proof that the control is causal.

{_markdown_table(['Control group', 'Estimable cells', 'Mean abs change', 'Median abs change', 'Maximum abs change'], group_rows)}

The per-cell winner count identifies `{overall_winner}` as the most frequent
source of Korean-coefficient movement in this run:

{_markdown_table(['Control group', 'Anchor/horizon wins'], winner_rows)}

The macro block contains seven controls in a sample of roughly 80 event days.
Its coefficient shifts can therefore reflect common-factor adjustment, but
also multicollinearity or over-parameterization. Treat the ranking as a
sensitivity diagnostic rather than evidence that the macro block is causal.

The regular-session-only comparison is:

{_markdown_table(['Control group', 'Regular cells', 'Mean abs change', 'Median abs change', 'Maximum abs change'], regular_rows)}

## Anchor-by-horizon comparison

`Macro`, `Korean macro`, `sector`, `liquidity`, and `microstructure` columns
report the absolute coefficient change defined above.

{_markdown_table(['Anchor', 'Horizon', 'Baseline beta KR', 'Macro', 'Korean macro', 'Sector', 'Liquidity', 'Microstructure', 'Largest change', 'Selected order', 'N'], headline_rows)}

## Baseline Granger inference

`Beta KR` is $\\widehat{{\\beta}}_0$, the estimated change in the future SKHY
log return associated with a one-unit increase in `kr_return_0`, conditional on
the selected lags and controls. Its sign gives the direction of the association.
`Beta p` is the two-sided HAC-robust p-value for
$H_0:\\beta_0=0$; it measures evidence against a zero current-Korean-return
coefficient, not the economic size of the coefficient. `Block p` is the
HAC-robust joint Wald p-value for all Korean-return terms retained by BIC.

{_markdown_table(['Anchor', 'Horizon', 'Beta KR', 'Beta p', 'Block p', 'Selected order', 'N'], baseline_rows)}

{inference_text}

## Availability

Unavailable cells: {unavailable_text}.

Dynamic-premarket cells are unavailable when full quote ticks are absent, no
candidate passes the predeclared liquidity filters, or the required model
variables are incomplete. No unavailable anchor is time- or price-filled.

## Interpretation limits

- All comparisons use a common sample and identical selected lag orders within
  an anchor/horizon cell.
- HAC inference is reported in the machine-readable results, but coefficient
  movement alone does not establish significance or causal mediation.
- The eight horizons overlap, the sample is small, and no multiple-testing
  correction is applied in this descriptive ablation ranking.
- Post-anchor trade/quote variables are excluded to prevent look-ahead bias.

Full estimates are stored only in
`models/results/granger_ablation_results.parquet`.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=GRANGER_PANEL_PATH)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--max-p", type=int, default=2)
    parser.add_argument("--max-q", type=int, default=2)
    parser.add_argument("--hac-maxlags", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_p not in (0, 1, 2) or args.max_q not in (0, 1, 2):
        raise SystemExit("max-p and max-q must each be 0, 1, or 2")
    if args.hac_maxlags < 0:
        raise SystemExit("hac-maxlags must be non-negative")

    panel = load_granger_panel(args.panel)
    results = run_ablations(
        panel,
        max_p=args.max_p,
        max_q=args.max_q,
        hac_maxlags=args.hac_maxlags,
    )
    args.results_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = args.results_dir / "granger_ablation_results.parquet"
    results.to_parquet(parquet_path, index=False, compression="zstd")
    write_model_design(MODEL_DOC_PATH, args.max_p, args.max_q)
    write_model_summary(results, SUMMARY_DOC_PATH)

    estimated = results[results["status"] == "estimated"]
    largest = estimated[
        estimated["largest_abs_change_in_cell"]
    ][["anchor_type", "horizon_minutes", "ablation", "abs_beta_change_from_baseline"]]
    print(f"Estimated specifications: {len(estimated):,}")
    print(f"Unavailable specifications: {(results['status'] != 'estimated').sum():,}")
    print("Largest coefficient change by anchor/horizon:")
    print(largest.to_string(index=False))
    print(f"Results: {parquet_path}")
    print(f"Design: {MODEL_DOC_PATH}")
    print(f"Summary: {SUMMARY_DOC_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
