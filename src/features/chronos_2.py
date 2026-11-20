"""Materialize the standalone Chronos-2 overnight feature table."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.config import PROJECT_ROOT, feature_output_path
from src.features.utils import build_overnight_features, write_feature_table


CHRONOS_FEATURE_PATH = PROJECT_ROOT / "data" / "features" / "overnight" / "part-00000.parquet"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=CHRONOS_FEATURE_PATH)
    args = parser.parse_args()

    output = feature_output_path(args.output)
    frame = build_overnight_features()
    write_feature_table(frame, output)
    print(f"Wrote {len(frame)} overnight feature rows to {output}")


if __name__ == "__main__":
    main()
