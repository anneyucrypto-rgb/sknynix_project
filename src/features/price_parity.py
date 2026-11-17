"""SKHY–000660 price-parity transformations."""

from __future__ import annotations

import numpy as np
import pandas as pd


DEFAULT_ORDINARY_SHARES_PER_ADS = 0.1


def theoretical_adr_usd(
    kr_price_krw: float | pd.Series,
    usdkrw: float | pd.Series,
    ordinary_shares_per_ads: float = DEFAULT_ORDINARY_SHARES_PER_ADS,
) -> float | pd.Series:
    """Return the theoretical USD ADS price from Korean shares and FX."""

    if ordinary_shares_per_ads <= 0:
        raise ValueError("ordinary_shares_per_ads must be positive")
    return kr_price_krw * ordinary_shares_per_ads / usdkrw


def log_kr_usd(
    kr_price_krw: float | pd.Series,
    usdkrw: float | pd.Series,
    ordinary_shares_per_ads: float = DEFAULT_ORDINARY_SHARES_PER_ADS,
) -> float | pd.Series:
    theoretical = theoretical_adr_usd(
        kr_price_krw, usdkrw, ordinary_shares_per_ads
    )
    return np.log(theoretical)


def equilibrium_error(
    log_adr: float | pd.Series,
    log_korean_usd: float | pd.Series,
    *,
    intercept: float = 0.0,
    theta: float = 1.0,
) -> float | pd.Series:
    return log_adr - intercept - theta * log_korean_usd


def add_parity_columns(
    frame: pd.DataFrame,
    *,
    adr_column: str,
    kr_column: str,
    fx_column: str,
    ordinary_shares_per_ads: float = DEFAULT_ORDINARY_SHARES_PER_ADS,
) -> pd.DataFrame:
    """Add theoretical price, logs, and parity spread with invalids as null."""

    output = frame.copy()
    valid = (
        output[adr_column].gt(0)
        & output[kr_column].gt(0)
        & output[fx_column].gt(0)
    )
    theoretical = theoretical_adr_usd(
        output[kr_column], output[fx_column], ordinary_shares_per_ads
    )
    output["theoretical_adr_usd"] = theoretical.where(valid)
    output["log_adr"] = np.log(output[adr_column]).where(valid)
    output["log_kr_usd"] = np.log(theoretical).where(valid)
    output["parity_spread"] = (
        output["log_adr"] - output["log_kr_usd"]
    ).where(valid)
    return output
