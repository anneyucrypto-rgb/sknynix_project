"""Dataset contract, uniqueness, and point-in-time validators."""

from __future__ import annotations

import pandas as pd

from src.features.contracts import CONTRACTS, DatasetContract


class DatasetValidationError(ValueError):
    pass


def validate_contract(
    frame: pd.DataFrame,
    contract: str | DatasetContract,
    *,
    allow_empty: bool = False,
) -> None:
    spec = CONTRACTS[contract] if isinstance(contract, str) else contract
    missing = set(spec.required_columns).difference(frame.columns)
    if missing:
        raise DatasetValidationError(
            f"{spec.name} is missing required columns: {sorted(missing)}"
        )
    if frame.empty and not allow_empty:
        raise DatasetValidationError(f"{spec.name} contains no observations")
    duplicated = frame.duplicated(list(spec.primary_key), keep=False)
    if duplicated.any():
        raise DatasetValidationError(
            f"{spec.name} has {int(duplicated.sum())} duplicate primary-key rows"
        )
    for column in spec.timestamp_columns:
        values = pd.to_datetime(frame[column], errors="coerce", utc=True)
        if frame[column].notna().any() and values.notna().sum() != frame[column].notna().sum():
            raise DatasetValidationError(f"{spec.name}.{column} has invalid timestamps")


def validate_no_lookahead(frame: pd.DataFrame, dataset: str) -> None:
    violations: list[str] = []
    if dataset == "daily":
        mask = frame["kr_available_at_utc"].notna() & (
            pd.to_datetime(frame["kr_available_at_utc"], utc=True)
            > pd.to_datetime(frame["us_close_ts_utc"], utc=True)
        )
        if mask.any():
            violations.append(f"{int(mask.sum())} Korean closes unavailable at U.S. close")
        fx = frame["fx_asof_ts_utc"].notna() & (
            pd.to_datetime(frame["fx_asof_ts_utc"], utc=True)
            > pd.to_datetime(frame["us_close_ts_utc"], utc=True)
        )
        if fx.any():
            violations.append(f"{int(fx.sum())} future FX observations")
    elif dataset == "us_5min":
        cutoff = pd.to_datetime(frame["bar_end_utc"], utc=True)
        for column in ("last_quote_ts", "fx_ts", "available"):
            if column in frame:
                observed = pd.to_datetime(frame[column], utc=True)
                bad = observed.notna() & (observed > cutoff)
                if bad.any():
                    violations.append(f"{int(bad.sum())} future {column} observations")
    elif dataset == "h2":
        bad = frame["valid_outcome"].fillna(False) & (
            pd.to_datetime(frame["endpoint_ts_utc"], utc=True)
            <= pd.to_datetime(frame["anchor_ts_utc"], utc=True)
        )
        if bad.any():
            violations.append(f"{int(bad.sum())} non-future H2 endpoints")
        if "outcome_ts" in frame:
            outcome = pd.to_datetime(frame["outcome_ts"], utc=True)
            endpoint = pd.to_datetime(frame["endpoint_ts_utc"], utc=True)
            future = frame["valid_outcome"].fillna(False) & (outcome > endpoint)
            stale = frame["valid_outcome"].fillna(False) & ((endpoint - outcome) > pd.Timedelta("60s"))
            if future.any() or stale.any():
                violations.append(f"{int((future | stale).sum())} invalid H2 endpoint matches")
    elif dataset == "h3":
        if frame["information_cutoff_utc"].isna().any():
            violations.append("H3 contains missing information cutoffs")
        us_to_kr = frame["direction"].eq("us_to_korea")
        if us_to_kr.any():
            target_open = pd.to_datetime(
                frame.loc[us_to_kr, "target_session_date"].astype(str) + " 09:00"
            ).dt.tz_localize("Asia/Seoul").dt.tz_convert("UTC")
            cutoff = pd.to_datetime(
                frame.loc[us_to_kr, "information_cutoff_utc"], utc=True
            )
            bad = cutoff >= target_open
            if bad.any():
                violations.append(f"{int(bad.sum())} U.S.-to-Korea predictors miss the target-session cutoff")
    if violations:
        raise DatasetValidationError(f"{dataset} look-ahead validation failed: " + "; ".join(violations))


def validation_summary(frame: pd.DataFrame) -> dict[str, object]:
    return {
        "rows": len(frame),
        "columns": len(frame.columns),
        "missing_fraction": float(frame.isna().mean().mean()) if len(frame) else 1.0,
        "duplicate_rows": int(frame.duplicated().sum()),
    }
