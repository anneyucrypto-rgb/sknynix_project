"""Diagnose multicollinearity within and among macro-control groups."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Final
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

try:
    from granger import GRANGER_PANEL_PATH
except ImportError:  # Support ``python -m models.feature_analysis``.
    from models.granger import GRANGER_PANEL_PATH


PROJECT_DIR: Final = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_PATH: Final = (
    PROJECT_DIR / "design" / "macro_multicollinearity.md"
)
ANCHOR_GROUPS: Final = {
    "premarket_dynamic": {
        "equity_market": ("spy_pre_event_return_5m", "kospi_return_latest"),
        "fx": ("usdkrw_return_latest",),
        "volatility": ("vix_level_lag1d", "vix_change_lag1d"),
        "rates": ("us_2y_yield_change_lag1d", "us_10y_yield_change_lag1d"),
        "korea_equity_market": (
            "kospi_5m_return_latest",
            "kospi200_5m_return_latest",
            "kosdaq150_5m_return_latest",
            "krx300_5m_return_latest",
            "kosdaq_5m_return_latest",
        ),
        "korea_semiconductor": (
            "korea_semiconductor_5m_return_latest",
            "korea_semiconductor_top10_5m_return_latest",
        ),
    },
    "regular_0930": {
        "equity_market": ("spy_pre_event_return_5m", "kospi_return_latest"),
        "fx": ("usdkrw_return_latest",),
        "volatility": ("vix_level_lag1d", "vix_change_lag1d"),
        "rates": ("us_2y_yield_change_lag1d", "us_10y_yield_change_lag1d"),
        "korea_equity_market": (
            "kospi_5m_return_latest",
            "kospi200_5m_return_latest",
            "kosdaq150_5m_return_latest",
            "krx300_5m_return_latest",
            "kosdaq_5m_return_latest",
        ),
        "korea_semiconductor": (
            "korea_semiconductor_5m_return_latest",
            "korea_semiconductor_top10_5m_return_latest",
        ),
    },
}


def _vif(values: pd.DataFrame) -> pd.Series:
    """Return VIFs without requiring a statsmodels helper API."""
    if values.shape[1] == 1:
        return pd.Series(1.0, index=values.columns, dtype=float)
    standardized = (values - values.mean()) / values.std(ddof=0)
    result: dict[str, float] = {}
    for column in standardized:
        y = standardized[column].to_numpy(dtype=float)
        others = standardized.drop(columns=column).to_numpy(dtype=float)
        design = np.column_stack([np.ones(len(others)), others])
        coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
        residual = y - design @ coefficients
        total_ss = float(np.dot(y - y.mean(), y - y.mean()))
        residual_ss = float(np.dot(residual, residual))
        r_squared = 1.0 - residual_ss / total_ss if total_ss > 0 else 1.0
        result[column] = np.inf if r_squared >= 1.0 - 1e-12 else 1.0 / (1.0 - r_squared)
    return pd.Series(result, dtype=float)


def _first_component(values: pd.DataFrame) -> pd.Series:
    """Build an oriented, standardized first-PC score for one feature group."""
    standardized = (values - values.mean()) / values.std(ddof=0)
    if standardized.shape[1] == 1:
        return standardized.iloc[:, 0]
    _, _, right_vectors = np.linalg.svd(standardized.to_numpy(), full_matrices=False)
    loadings = right_vectors[0]
    if loadings[0] < 0:
        loadings = -loadings
    return pd.Series(standardized.to_numpy() @ loadings, index=values.index)


def _condition_number(values: pd.DataFrame) -> float:
    standardized = (values - values.mean()) / values.std(ddof=0)
    singular_values = np.linalg.svd(standardized.to_numpy(), compute_uv=False)
    if len(singular_values) == 0 or singular_values[-1] <= 1e-12:
        return np.inf
    return float(singular_values[0] / singular_values[-1])


def _iterative_selection(
    values: pd.DataFrame,
    feature_groups: dict[str, str],
    threshold: float,
) -> tuple[list[str], list[dict[str, object]]]:
    """Drop the worst VIF while retaining at least one feature per group."""
    retained = list(values.columns)
    steps: list[dict[str, object]] = []
    while len(retained) > 1:
        vifs = _vif(values[retained])
        group_counts = pd.Series([feature_groups[name] for name in retained]).value_counts()
        removable = [
            name for name in retained if group_counts[feature_groups[name]] > 1
        ]
        if not removable:
            break
        candidate = max(removable, key=lambda name: float(vifs[name]))
        candidate_vif = float(vifs[candidate])
        if np.isfinite(candidate_vif) and candidate_vif <= threshold:
            break
        retained.remove(candidate)
        steps.append(
            {
                "step": len(steps) + 1,
                "dropped_feature": candidate,
                "group": feature_groups[candidate],
                "vif_at_drop": candidate_vif,
            }
        )
    return retained, steps


def analyze_anchor(
    panel: pd.DataFrame,
    anchor: str,
    groups: dict[str, tuple[str, ...]],
    threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    columns = [feature for features in groups.values() for feature in features]
    missing = sorted(set(columns).difference(panel.columns))
    if missing:
        raise ValueError(f"Panel is missing macro columns for {anchor}: {missing}")
    sample = panel.loc[panel["anchor_type"] == anchor, columns].dropna().astype(float)
    if len(sample) <= len(columns) + 1:
        raise ValueError(
            f"{anchor} has {len(sample)} complete rows; more than {len(columns) + 1} are required"
        )
    zero_variance = [name for name in columns if sample[name].nunique() <= 1]
    if zero_variance:
        raise ValueError(f"Constant macro columns for {anchor}: {zero_variance}")

    feature_to_group = {
        feature: group for group, features in groups.items() for feature in features
    }
    feature_vif = (
        _vif(sample).rename("vif").rename_axis("feature").reset_index()
    )
    feature_vif.insert(0, "anchor_type", anchor)
    feature_vif.insert(2, "group", feature_vif["feature"].map(feature_to_group))
    feature_vif["exceeds_threshold"] = feature_vif["vif"] > threshold

    composites = pd.DataFrame(
        {group: _first_component(sample[list(features)]) for group, features in groups.items()}
    )
    group_vif = (
        _vif(composites).rename("vif").rename_axis("group").reset_index()
    )
    group_vif.insert(0, "anchor_type", anchor)
    group_vif["exceeds_threshold"] = group_vif["vif"] > threshold

    correlation = sample.corr().rename_axis("feature").reset_index()
    correlation.insert(0, "anchor_type", anchor)
    retained, steps = _iterative_selection(sample, feature_to_group, threshold)
    selection = pd.DataFrame(steps)
    if selection.empty:
        selection = pd.DataFrame(columns=["step", "dropped_feature", "group", "vif_at_drop"])
    selection.insert(0, "anchor_type", anchor)
    selection["retained_features_after_selection"] = ", ".join(retained)

    diagnostics = {
        "anchor_type": anchor,
        "n_complete": len(sample),
        "n_features": len(columns),
        "condition_number": _condition_number(sample),
        "max_feature_vif": float(feature_vif["vif"].max()),
        "max_group_vif": float(group_vif["vif"].max()),
        "retained_features": ", ".join(retained),
    }
    return feature_vif, group_vif, correlation, selection, diagnostics


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "No rows."
    display = frame.copy()
    for column in display.select_dtypes(include=["float"]).columns:
        display[column] = display[column].map(
            lambda value: "inf" if np.isinf(value) else f"{value:.3f}"
        )
    headers = [str(column) for column in display.columns]
    rows = [
        [str(value).replace("|", "\\|") for value in row]
        for row in display.itertuples(index=False, name=None)
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def write_report(
    path: Path,
    feature_vif: pd.DataFrame,
    group_vif: pd.DataFrame,
    correlations: pd.DataFrame,
    selection: pd.DataFrame,
    diagnostics: pd.DataFrame,
    unavailable: pd.DataFrame,
    threshold: float,
) -> None:
    generated = datetime.now(ZoneInfo("Asia/Singapore")).strftime("%Y-%m-%d %H:%M:%S SGT")
    feature_breaches = int(feature_vif["exceeds_threshold"].sum())
    group_breaches = int(group_vif["exceeds_threshold"].sum())
    if feature_breaches == 0 and group_breaches == 0:
        conclusion = (
            "No feature-level or group-level VIF exceeds the threshold. The "
            "current macro block does not require VIF-driven reduction."
        )
    else:
        conclusion = (
            f"The analysis identifies {feature_breaches} feature-level and "
            f"{group_breaches} group-level threshold breaches; review the "
            "iterative reduction before fitting the macro block."
        )
    report = f"""# Macro-feature multicollinearity analysis

Generated: {generated}

The feature-level table estimates variance inflation factors (VIFs) from the
complete-case macro block separately at each event anchor. The group-level
table first reduces each conceptual group to its standardized first principal
component, then calculates VIF across those group scores. The diagnostic
threshold is {threshold:g}; it is a screening rule, not a mechanical proof that
a variable must be removed.

## Conclusion

{conclusion}

## Anchor diagnostics

{_markdown_table(diagnostics)}

## Unavailable anchors

{_markdown_table(unavailable)}

## Feature-level VIF

{_markdown_table(feature_vif)}

## Group-level VIF

{_markdown_table(group_vif)}

## Feature correlations

The following Pearson correlation matrices use the same complete-case samples
as the VIF calculations.

{_markdown_table(correlations)}

## Iterative reduction

The reduction drops the highest-VIF feature above the threshold while retaining
at least one feature from every conceptual group.

{_markdown_table(selection)}
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=GRANGER_PANEL_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--vif-threshold", type=float, default=10.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.vif_threshold <= 1:
        raise SystemExit("vif-threshold must be greater than 1")
    panel = pd.read_parquet(args.panel)
    outputs = []
    unavailable_rows = []
    for anchor, groups in ANCHOR_GROUPS.items():
        try:
            outputs.append(analyze_anchor(panel, anchor, groups, args.vif_threshold))
        except ValueError as exc:
            unavailable_rows.append({"anchor_type": anchor, "reason": str(exc)})
    if not outputs:
        raise ValueError("No anchor has enough complete macro data for VIF analysis")
    feature_vif = pd.concat([output[0] for output in outputs], ignore_index=True)
    group_vif = pd.concat([output[1] for output in outputs], ignore_index=True)
    correlations = pd.concat([output[2] for output in outputs], ignore_index=True)
    selection = pd.concat([output[3] for output in outputs], ignore_index=True)
    diagnostics = pd.DataFrame([output[4] for output in outputs])
    unavailable = pd.DataFrame(unavailable_rows, columns=["anchor_type", "reason"])

    write_report(
        args.output,
        feature_vif,
        group_vif,
        correlations,
        selection,
        diagnostics,
        unavailable,
        args.vif_threshold,
    )
    print(f"Report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
