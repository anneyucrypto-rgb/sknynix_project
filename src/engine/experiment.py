"""Validated, hashable research experiment specification."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
import hashlib
import json

from src.engine.control_registry import validate_controls


@dataclass(frozen=True)
class ExperimentSpec:
    hypothesis: str
    estimator: str
    horizons: tuple[int, ...] = (30,)
    controls: tuple[str, ...] = ("us_soxx_return", "kr_sector_return")
    max_lags: int = 5
    significance_level: float = 0.05
    selection_criterion: str = "bic"
    inference: str = "hac"
    diagnostics: str = "full"
    start_date: date | None = None
    end_date: date | None = None
    training_start_date: date | None = None
    training_end_date: date | None = None
    h1_model_id: str = "theory_parity"
    h1_intercept: float | None = None
    h1_theta: float | None = None
    materialize_features: bool = True
    generate_markdown: bool = True

    def validate(self) -> None:
        if self.hypothesis.upper() not in {"H1", "H2", "H3"}:
            raise ValueError("hypothesis must be H1, H2, or H3")
        if not self.estimator.strip():
            raise ValueError("estimator is required")
        if not self.horizons or any(value <= 0 for value in self.horizons):
            raise ValueError("horizons must be positive minute intervals")
        if self.max_lags < 1:
            raise ValueError("max_lags must be positive")
        if not 0 < self.significance_level < 1:
            raise ValueError("significance_level must be between zero and one")
        if self.start_date and self.end_date and self.start_date > self.end_date:
            raise ValueError("start_date must not follow end_date")
        if self.training_start_date and self.training_end_date and self.training_start_date > self.training_end_date:
            raise ValueError("training_start_date must not follow training_end_date")
        if self.training_end_date and self.start_date and self.training_end_date >= self.start_date:
            raise ValueError("training must end before evaluation starts")
        validate_controls(self.controls)
        if (
            self.h1_model_id != "theory_parity"
            and (self.h1_intercept is None or self.h1_theta is None)
        ):
            raise ValueError("an estimated h1_model_id requires h1_intercept and h1_theta")

    def canonical_json(self) -> str:
        self.validate()
        payload = asdict(self)
        payload["start_date"] = self.start_date.isoformat() if self.start_date else None
        payload["end_date"] = self.end_date.isoformat() if self.end_date else None
        payload["training_start_date"] = self.training_start_date.isoformat() if self.training_start_date else None
        payload["training_end_date"] = self.training_end_date.isoformat() if self.training_end_date else None
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(self.canonical_json().encode()).hexdigest()[:16]

    def new_run_id(self) -> str:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        return f"{stamp}-{self.hypothesis.lower()}-{self.fingerprint[:8]}"


@dataclass(frozen=True)
class ResearchConfig:
    adr_ticker: str = "SKHY"
    korea_ticker: str = "000660"
    ordinary_shares_per_ads: float = 0.1
    default_horizons: tuple[int, ...] = (30,)
