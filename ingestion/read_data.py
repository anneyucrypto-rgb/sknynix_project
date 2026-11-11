"""Display the two processed research panels."""

from pathlib import Path

import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
DATASETS = {
    "granger": PROJECT / "data" / "processed" / "granger_panel.parquet",
    "arbitrage": PROJECT / "data" / "processed" / "arbitrage_panel_1min.parquet",
}


def main() -> None:
    for name, path in DATASETS.items():
        frame = pd.read_parquet(path)
        print(f"{name}: {len(frame):,} rows x {len(frame.columns):,} columns")
        print(frame.head())


if __name__ == "__main__":
    main()
