"""VARX models and their Korean daily-price input loader."""

from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
KOREAN_DAILY_PATH = PROJECT_DIR / "data" / "raw" / "korea" / "000660_daily.parquet"
GRANGER_PANEL_PATH = PROJECT_DIR / "data" / "processed" / "granger_panel.parquet"
REQUIRED_KOREAN_COLUMNS = {
    "ticker",
    "date",
    "open",
    "high",
    "low",
    "close",
    "adjusted_close",
    "volume",
}


def load_korean_daily_prices(path: Path = KOREAN_DAILY_PATH) -> pd.DataFrame:
    """Load, validate, and chronologically order daily 000660.KS prices."""
    frame = pd.read_parquet(path)
    missing = REQUIRED_KOREAN_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Korean daily-price file is missing columns: {sorted(missing)}")
    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    return frame.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)


def load_granger_panel(path: Path = GRANGER_PANEL_PATH) -> pd.DataFrame:
    """Load the processed event-level Granger/VARX feature panel."""
    frame = pd.read_parquet(path)
    frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True)
    return frame.sort_values(["event_date_ny", "anchor_type"]).reset_index(drop=True)
