"""Stable control definitions, including explicitly labelled Korean proxies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ControlSpec:
    name: str
    sql_path: str | None
    frequency: str
    join_method: Literal["asof", "previous_session", "event_window"]
    source_column: str
    max_age_minutes: int | None = None
    min_coverage: float = 0.80
    is_proxy: bool = False
    description: str = ""


CONTROLS: dict[str, ControlSpec] = {
    "us_soxx_return": ControlSpec("us_soxx_return", "sql/controls/us_sector_return.sql", "intraday", "asof", "us_sector_return", 5, description="SOXX return known at cutoff"),
    "kr_sector_return": ControlSpec("kr_sector_return", "sql/controls/kr_sector_return.sql", "daily", "previous_session", "kr_sector_return_known", description="Korean semiconductor sector return known by cutoff"),
    "us_2y_yield_change": ControlSpec("us_2y_yield_change", "sql/controls/us_yields.sql", "daily", "asof", "us_2y_yield_change", 10_080),
    "us_10y_yield_change": ControlSpec("us_10y_yield_change", "sql/controls/us_yields.sql", "daily", "asof", "us_10y_yield_change", 10_080),
    "us_inflation_yoy": ControlSpec("us_inflation_yoy", "sql/controls/economic_surprises.sql", "monthly", "asof", "us_inflation_yoy", 89_280),
    "korea_equity_risk_proxy": ControlSpec("korea_equity_risk_proxy", None, "daily", "previous_session", "korea_equity_risk_proxy", is_proxy=True, description="EWY market proxy; not a direct Korean macro observation"),
    "korea_currency_policy_expectations_proxy": ControlSpec("korea_currency_policy_expectations_proxy", None, "daily", "asof", "korea_currency_policy_expectations_proxy", 10_080, is_proxy=True, description="USDKRW market proxy; not a Bank of Korea policy observation"),
}


def validate_controls(names: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    unknown = set(names).difference(CONTROLS)
    if unknown:
        raise ValueError(f"Unknown controls: {sorted(unknown)}")
    return tuple(dict.fromkeys(names))
