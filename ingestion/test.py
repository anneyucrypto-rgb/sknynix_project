"""Audit SKHY panel freshness and list Korean index signal candidates."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pandas as pd
import requests


PROJECT_DIR: Final = Path(__file__).resolve().parents[1]
INGESTION_DIR: Final = Path(__file__).resolve().parent
RAW_DIR: Final = PROJECT_DIR / "data" / "raw"
PROCESSED_DIR: Final = PROJECT_DIR / "data" / "processed"
SECRET_FILE: Final = INGESTION_DIR / ".env"
MASSIVE_TICKERS_URL: Final = "https://api.massive.com/v3/reference/tickers"


@dataclass(frozen=True)
class KoreanSignalCandidate:
    name: str
    role: str
    symbol: str
    source: str
    priority: str
    note: str


# A bounded research universe. KRX calculates more than one thousand indices,
# so an unfiltered claim to list every Korean index would not be useful.
CURATED_CANDIDATES: Final = (
    KoreanSignalCandidate("KOSPI Composite", "broad market", "^KS11", "Yahoo", "core", "Current project macro baseline."),
    KoreanSignalCandidate("KOSPI 200", "large-cap market", "^KS200", "Yahoo/KRX", "core", "Official benchmark; direct Yahoo history can be sparse."),
    KoreanSignalCandidate("KRX 300", "cross-market broad", "", "KRX/vendor", "core", "Broad KOSPI and KOSDAQ representation."),
    KoreanSignalCandidate("KOSDAQ Composite", "growth/technology", "^KQ11", "Yahoo", "core", "Usable Korean growth and technology-risk series."),
    KoreanSignalCandidate("KOSDAQ 150", "growth large-cap", "", "KRX/vendor", "secondary", "Technology-heavy KOSDAQ benchmark."),
    KoreanSignalCandidate("KRX Semiconductor", "semiconductor sector", "", "KRX/vendor", "core", "Most specific domestic industry factor for SK hynix."),
    KoreanSignalCandidate("KOSPI Electrical & Electronic Equipment", "electronics sector", "", "KRX/vendor", "core", "Broad electronics and semiconductor exposure."),
    KoreanSignalCandidate("KOSPI 200 Information Technology", "large-cap IT", "", "KRX/vendor", "core", "Closer industry control than broad KOSPI."),
    KoreanSignalCandidate("KRX 300 Information Technology", "cross-market IT", "", "KRX/vendor", "secondary", "IT factor spanning both Korean boards."),
    KoreanSignalCandidate("V-KOSPI 200", "domestic volatility", "", "KRX/vendor", "core", "Korean equity uncertainty and risk regime."),
    KoreanSignalCandidate("KRX Mid 200", "market breadth", "", "KRX/vendor", "secondary", "Separates market breadth from mega-cap leadership."),
    KoreanSignalCandidate("Korea Value-up", "policy/corporate reform", "", "KRX/vendor", "secondary", "Policy-sensitive Korean large-company factor."),
    KoreanSignalCandidate("Korea-Taiwan IT Premier", "regional IT cycle", "", "KRX/vendor", "secondary", "Linked Korean and Taiwanese technology cycle."),
    KoreanSignalCandidate("KRX CSI Korea-China Semiconductor", "regional semiconductor", "", "KRX/vendor", "secondary", "Cross-border semiconductor robustness factor."),
    KoreanSignalCandidate("KODEX 200", "KOSPI 200 proxy", "069500.KS", "Yahoo ETF", "fallback", "Verified downloadable ETF proxy."),
    KoreanSignalCandidate("KODEX KOSDAQ 150", "KOSDAQ 150 proxy", "229200.KS", "Yahoo ETF", "fallback", "Verified downloadable ETF proxy."),
    KoreanSignalCandidate("KODEX Semiconductor", "semiconductor proxy", "091160.KS", "Yahoo ETF", "fallback", "Preferred downloadable semiconductor proxy."),
    KoreanSignalCandidate("TIGER Fn Semiconductor TOP10", "semiconductor leaders", "396500.KS", "Yahoo ETF", "fallback", "Concentrated semiconductor-leader proxy."),
)


def load_secret(path: Path = SECRET_FILE) -> str:
    """Load MASSIVE_API_KEY from the environment or ingestion/.env."""
    key = os.getenv("MASSIVE_API_KEY", "").strip()
    if key:
        return key
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            name, value = stripped.split("=", 1)
            if name.strip() == "MASSIVE_API_KEY":
                return value.strip().strip("\"'")
    raise RuntimeError(
        f"MASSIVE_API_KEY is missing from the environment and {path}"
    )


def assert_processed_panel_is_current() -> None:
    """Fail when processed event dates lag the current raw SKHY bar file."""
    raw = pd.read_parquet(
        RAW_DIR / "us" / "SKHY_5min.parquet", columns=["timestamp"]
    )
    processed = pd.read_parquet(
        PROCESSED_DIR / "granger_panel.parquet", columns=["event_date_ny"]
    )
    raw_dates = set(
        pd.to_datetime(raw["timestamp"], utc=True)
        .dt.tz_convert("America/New_York")
        .dt.date
    )
    processed_dates = set(pd.to_datetime(processed["event_date_ny"]).dt.date)
    if raw_dates != processed_dates:
        missing = sorted(raw_dates.difference(processed_dates))
        extra = sorted(processed_dates.difference(raw_dates))
        raise AssertionError(
            "Processed Granger dates are stale. Rerun ingestion/features.py. "
            f"Missing dates: {missing}; extra dates: {extra}"
        )
    print(
        f"Panel freshness: OK ({len(raw_dates)} SKHY trading days, "
        f"{min(raw_dates)} through {max(raw_dates)})"
    )


def fetch_massive_korea_indices(api_key: str) -> pd.DataFrame:
    """Return deduplicated Massive indices that directly represent Korea."""
    response = requests.get(
        MASSIVE_TICKERS_URL,
        params={
            "market": "indices",
            "search": "Korea",
            "active": "true",
            "limit": 1000,
            "sort": "ticker",
            "order": "asc",
            "apiKey": api_key,
        },
        timeout=30,
    )
    response.raise_for_status()
    records = []
    for item in response.json().get("results", []):
        name = str(item.get("name", ""))
        lower = name.lower()
        if "korea" not in lower or "ex korea" in lower or "plus korea" in lower:
            continue
        if "target date" in lower or "sovereign" in lower:
            continue
        if any(term in lower for term in ("health", "financial", "real estate", "consumer", "energy", "utilities", "industrials", "materials")):
            continue
        role = "technology" if "technology" in lower else "country/size/style"
        records.append(
            {
                "ticker": item.get("ticker", ""),
                "name": name.replace("&amp;", "&"),
                "role": role,
            }
        )
    if not records:
        return pd.DataFrame(columns=["ticker", "name", "role"])
    return (
        pd.DataFrame.from_records(records)
        .drop_duplicates("ticker")
        .sort_values(["role", "ticker"])
        .reset_index(drop=True)
    )


def print_curated_candidates() -> None:
    frame = pd.DataFrame(
        [candidate.__dict__ for candidate in CURATED_CANDIDATES]
    )
    print("\nRecommended Korean macro/sector index universe for SKHY:")
    print(frame.to_string(index=False))


def main() -> int:
    assert_processed_panel_is_current()
    print_curated_candidates()
    massive = fetch_massive_korea_indices(load_secret())
    print("\nDirectly available Korea country/technology indices in Massive:")
    print(massive.to_string(index=False) if not massive.empty else "None found")
    print(
        "\nRecommended starting controls: ^KS11 (broad market), ^KQ11 "
        "(growth/technology), a KRX semiconductor index or 091160.KS proxy, "
        "and V-KOSPI 200 when a licensed KRX series is available."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
