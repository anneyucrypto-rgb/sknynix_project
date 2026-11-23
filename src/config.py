"""Typed project settings shared by ingestion and research pipelines."""

from __future__ import annotations

import os
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final


PROJECT_ROOT: Final = Path(__file__).resolve().parents[1]


def feature_output_path(path: Path, project_root: Path = PROJECT_ROOT) -> Path:
    """Resolve a table output beneath the project feature directory."""
    output = path.resolve()
    feature_root = (project_root / "data" / "features").resolve()
    if output == feature_root or not output.is_relative_to(feature_root):
        raise ValueError(f"Table outputs must be under {feature_root}: {path}")
    return output


@dataclass(frozen=True)
class PathSettings:
    """Canonical project paths; raw datasets are Parquet directories."""

    project_root: Path = PROJECT_ROOT

    @property
    def data(self) -> Path:
        return self.project_root / "data"

    @property
    def raw(self) -> Path:
        return self.data / "raw"

    @property
    def features(self) -> Path:
        return self.data / "features"

    def feature_run(self, run_id: str) -> Path:
        return self.features / "experiments" / f"run_id={run_id}"

    def raw_part(self, *parts: str) -> Path:
        return self.raw.joinpath(*parts, "part-00000.parquet")


@dataclass(frozen=True)
class MassiveSettings:
    base_url: str = "https://api.massive.com"
    api_key_env: str = "MASSIVE_API_KEY"
    ticker: str = "SKHY"
    control_tickers: tuple[str, ...] = ("SPY", "SOXX")
    korean_proxy_tickers: tuple[tuple[str, str], ...] = (
        ("EWY", "korea_equity_risk"),
        ("FLKR", "korea_equity_robustness"),
        ("C:USDKRW", "korea_currency_policy_expectations"),
    )
    bar_minutes: tuple[int, ...] = (1, 5)
    timeout_seconds: float = 60.0
    max_retries: int = 6


@dataclass(frozen=True)
class KoreanMarketSettings:
    daily_symbols: tuple[tuple[str, str], ...] = (
        ("000660.KS", "000660_daily"),
        ("^KS11", "KOSPI_daily"),
        ("^KQ11", "KOSDAQ_daily"),
        ("091160.KS", "KOREA_SEMICONDUCTOR_daily"),
    )


@dataclass(frozen=True)
class FxSettings:
    massive_ticker: str = "C:USDKRW"
    dataset_stem: str = "USDKRW"
    bar_minutes: tuple[int, ...] = (1, 5)


@dataclass(frozen=True)
class EconomicSettings:
    massive_us_endpoints: tuple[tuple[str, tuple[str, ...], str], ...] = (
        (
            "/fed/v1/treasury-yields",
            ("date", "yield_2_year", "yield_10_year"),
            "treasury/US_TREASURY_daily",
        ),
        (
            "/fed/v1/inflation",
            ("date", "cpi", "cpi_core", "cpi_year_over_year", "pce", "pce_core"),
            "inflation/US_INFLATION_monthly",
        ),
        (
            "/fed/v1/funding-conditions",
            ("date",),
            "policy/US_FUNDING_CONDITIONS_daily",
        ),
    )


@dataclass(frozen=True)
class IngestionSettings:
    paths: PathSettings = field(default_factory=PathSettings)
    massive: MassiveSettings = field(default_factory=MassiveSettings)
    korean: KoreanMarketSettings = field(default_factory=KoreanMarketSettings)
    fx: FxSettings = field(default_factory=FxSettings)
    economic: EconomicSettings = field(default_factory=EconomicSettings)


def read_secret(name: str, *, required: bool = True) -> str | None:
    """Read a secret from the environment or a local uncommitted env file."""

    value = os.getenv(name, "").strip()
    if value:
        return value
    for path in (PROJECT_ROOT / ".env",):
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, candidate = stripped.split("=", 1)
            if key.strip() == name:
                # Respect quoted values and ignore comments outside quotes, as
                # standard dotenv parsers do. For example:
                # MASSIVE_API_KEY="..."  # account dashboard
                lexer = shlex.shlex(candidate, posix=True)
                lexer.whitespace_split = True
                lexer.commenters = "#"
                value = " ".join(lexer).strip()
                if value:
                    return value
    if required:
        raise RuntimeError(f"{name} is required in the environment or a local .env file")
    return None
