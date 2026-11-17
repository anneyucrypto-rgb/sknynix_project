"""Registry for the six reproducible research datasets."""

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
    version: str = "v2"


FEATURES: dict[str, FeatureSpec] = {
    "daily": FeatureSpec("daily", None, "paired U.S. session", ("raw_skhy_1min", "raw_kr_daily", "raw_fx_1min", "raw_soxx_1min", "raw_kr_sector_daily"), "build_daily"),
    "us_5min": FeatureSpec("us_5min", None, "completed SKHY five-minute interval", ("raw_skhy_5min", "raw_skhy_quotes", "raw_fx_1min", "raw_kr_daily"), "build_us_5min"),
    "anchors": FeatureSpec("anchors", None, "U.S. date by anchor type", ("us_5min",), "build_anchors", "src.features.anchor_selector.select_anchors"),
    "h1": FeatureSpec("h1", None, "eligible paired daily observation", ("daily",), "build_h1"),
    "h2": FeatureSpec("h2", None, "date by anchor and horizon", ("anchors", "us_5min", "daily", "raw_skhy_1min"), "build_h2"),
    "h3": FeatureSpec("h3", None, "directional target observation", ("h2", "daily"), "build_h3"),
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
