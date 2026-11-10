from pathlib import Path

import pandas as pd

PROJECT = Path(r"C:\Users\anney\OneDrive\Documents\GitHub\sknynix_project")

datasets = {
    "SKHY": PROJECT / "data/us/SKHY_5min.parquet",
    "SPY": PROJECT / "data/us/SPY_5min.parquet",
    "SOXX": PROJECT / "data/us/SOXX_5min.parquet",
    "QQQ": PROJECT / "data/us/QQQ_5min.parquet",
    "USDKRW": PROJECT / "data/fx/USDKRW_5min.parquet",
}

frames = {
    name: pd.read_parquet(path)
    for name, path in datasets.items()
}

for name, frame in frames.items():
    print(name, frame.shape)
    print(frame.head())