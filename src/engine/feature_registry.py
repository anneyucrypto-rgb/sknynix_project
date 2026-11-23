"""Registry for reproducible research datasets."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    sql_path: str | None
    grain: str
    dependencies: tuple[str, ...]
    builder: str
    postprocessor: str | None = None
    version: str = "v6"


FEATURES: dict[str, FeatureSpec] = {
    "daily": FeatureSpec("daily", None, "paired U.S. session", ("raw_skhy_1min", "raw_kr_daily", "raw_fx_1min", "raw_soxx_1min", "raw_kr_sector_daily"), "build_daily"),
    "us_5min": FeatureSpec("us_5min", None, "completed SKHY five-minute interval", ("raw_skhy_5min", "raw_skhy_quotes", "raw_fx_1min", "raw_kr_daily"), "build_us_5min"),
    "h1": FeatureSpec("h1", None, "daily ARDL--UECM observation", ("daily",), "build_h1"),
    "h2": FeatureSpec("h2", None, "date by fixed thirty-minute response window", ("daily", "raw_skhy_1min", "raw_soxx_1min", "raw_fx_1min", "raw_kr_daily"), "build_h2"),
    "h3": FeatureSpec("h3", None, "U.S. close to next Korean session", ("daily",), "build_h3", version="v7"),
}


def get_feature(name: str) -> FeatureSpec:
    try:
        return FEATURES[name]
    except KeyError as exc:
        raise KeyError(f"Unknown feature {name!r}; expected one of {sorted(FEATURES)}") from exc


def register_feature(spec: FeatureSpec, *, replace: bool = False) -> None:
    if spec.name in FEATURES and not replace:
        raise ValueError(f"Feature already registered: {spec.name}")
    FEATURES[spec.name] = spec
