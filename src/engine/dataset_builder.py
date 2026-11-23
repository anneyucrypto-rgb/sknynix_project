"""Point-in-time builders for the six canonical research datasets."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import time
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.config import PathSettings
from src.engine.cache_manager import CacheManager, snapshot_hash
from src.engine.control_registry import CONTROLS, validate_controls
from src.engine.dependency_resolver import resolve_dependencies
from src.engine.duckdb_engine import DuckDBEngine
from src.engine.experiment import ExperimentSpec
from src.engine.feature_registry import FEATURES
from src.features.utils import (
    CONTRACTS,
    add_parity_columns,
    validate_contract,
    validate_no_lookahead,
)


NY = "America/New_York"
SEOUL = "Asia/Seoul"


def _utc_ns(values: object):
    """Normalize Arrow/DuckDB timestamp units for pandas as-of joins."""

    return pd.to_datetime(values, utc=True).astype("datetime64[ns, UTC]")


def _normalize_contract_timestamps(
    frame: pd.DataFrame, dataset: str
) -> pd.DataFrame:
    """Normalize cached Parquet timestamps before pandas joins/comparisons."""

    output = frame.copy()
    for column in CONTRACTS[dataset].timestamp_columns:
        if column in output:
            output[column] = _utc_ns(output[column])
    return output


@dataclass(frozen=True)
class DatasetBuildResult:
    name: str
    frame: pd.DataFrame
    output_path: Path | None
    cached: bool = False


class DatasetBuilder:
    """Build datasets with explicit cutoffs and optional Parquet materialization."""

    def __init__(self, paths: PathSettings | None = None, engine: DuckDBEngine | None = None) -> None:
        self.paths = paths or PathSettings()
        self.engine = engine or DuckDBEngine(self.paths.duckdb)
        self.cache = CacheManager(self.engine, self.paths)
        self.frames: dict[str, pd.DataFrame] = {}

    def close(self) -> None:
        self.engine.close()

    def __enter__(self) -> "DatasetBuilder":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _glob(self, *parts: str) -> Path:
        return self.paths.raw.joinpath(*parts)

    def _raw_source_paths(self, dataset: str) -> tuple[Path, ...]:
        mapping = {
            "raw_skhy_1min": self._glob("us", "market", "SKHY_1min"),
            "raw_skhy_5min": self._glob("us", "market", "SKHY_5min"),
            "raw_skhy_quotes": self._glob("us", "market", "SKHY_quotes"),
            "raw_kr_daily": self._glob("korea", "market", "000660_daily"),
            "raw_kr_sector_daily": self._glob("korea", "market", "sector_daily"),
            "raw_fx_1min": self._glob("fx", "USDKRW_1min"),
            "raw_soxx_1min": self._glob("us", "market", "SOXX_1min"),
        }
        found: set[Path] = set()

        def collect(name: str) -> None:
            for dependency in FEATURES[name].dependencies:
                if dependency in FEATURES:
                    collect(dependency)
                else:
                    found.add(mapping[dependency])

        collect(dataset)
        return tuple(sorted(found, key=str))

    @staticmethod
    def _parquet(path: Path) -> pd.DataFrame:
        files = sorted(path.rglob("*.parquet")) if path.is_dir() else [path]
        if not files:
            raise FileNotFoundError(f"No Parquet data found under {path}")
        return pd.concat((pd.read_parquet(file) for file in files), ignore_index=True)

    @staticmethod
    def _utc(frame: pd.DataFrame, column: str = "timestamp") -> pd.DataFrame:
        result = frame.copy()
        result[column] = _utc_ns(result[column])
        return result.sort_values(column)

    @staticmethod
    def _regular_close(bars: pd.DataFrame, prefix: str) -> pd.DataFrame:
        bars = DatasetBuilder._utc(bars)
        local = bars["timestamp"].dt.tz_convert(NY)
        minute = local.dt.hour * 60 + local.dt.minute
        regular_mask = minute.between(570, 959)
        regular = bars[regular_mask].copy()
        regular["date_us"] = local[regular_mask].dt.date
        result = regular.sort_values("timestamp").groupby("date_us", as_index=False).tail(1)
        result = result.copy()
        result["timestamp"] = result["timestamp"] + pd.Timedelta(minutes=1)
        return result[["date_us", "timestamp", "close"]].rename(columns={"timestamp": f"{prefix}_ts_utc", "close": f"{prefix}_close"})

    def build_daily(self) -> pd.DataFrame:
        adr = self._regular_close(self._parquet(self._glob("us", "market", "SKHY_1min")), "us_close")
        soxx = self._regular_close(self._parquet(self._glob("us", "market", "SOXX_1min")), "sector")
        soxx["us_sector_return"] = np.log(soxx["sector_close"]).diff()

        kr = self._parquet(self._glob("korea", "market", "000660_daily")).copy()
        kr["kr_session_date"] = pd.to_datetime(kr["date"]).dt.date
        kr["kr_close_krw"] = kr.get("adjusted_close", kr["close"])
        kr["kr_available_at_utc"] = _utc_ns(
            pd.to_datetime(kr["kr_session_date"].astype(str) + " 15:30")
            .dt.tz_localize(SEOUL)
        )
        kr["kr_close_ts_utc"] = kr["kr_available_at_utc"]
        kr["kr_daily_return_known"] = np.log(kr["kr_close_krw"]).diff()
        kr = kr.sort_values("kr_available_at_utc")

        fx = self._utc(self._parquet(self._glob("fx", "USDKRW_1min")))
        fx["timestamp"] = fx["timestamp"] + pd.Timedelta(minutes=1)
        fx = fx[["timestamp", "close"]].rename(columns={"timestamp": "fx_asof_ts_utc", "close": "usdkrw_asof"})

        daily = adr.rename(columns={"us_close_close": "adr_close_usd"}).sort_values("us_close_ts_utc")
        daily = pd.merge_asof(daily, kr[["kr_session_date", "kr_close_ts_utc", "kr_available_at_utc", "kr_close_krw", "kr_daily_return_known"]], left_on="us_close_ts_utc", right_on="kr_available_at_utc", direction="backward")
        daily = pd.merge_asof(daily, fx, left_on="us_close_ts_utc", right_on="fx_asof_ts_utc", direction="backward", tolerance=pd.Timedelta("60min"))
        daily = daily.merge(soxx[["date_us", "us_sector_return"]], on="date_us", how="left")

        sector_path = self._glob("korea", "market", "sector_daily")
        if list(sector_path.rglob("*.parquet")):
            sector = self._parquet(sector_path)
            sector["date"] = pd.to_datetime(sector["date"]).dt.date
            price = sector.get("adjusted_close", sector["close"])
            sector["kr_sector_return_known"] = np.log(price).groupby(sector.get("ticker", pd.Series("sector", index=sector.index))).diff()
            sector = sector.sort_values("date").drop_duplicates("date", keep="last")
            daily = daily.merge(sector[["date", "kr_sector_return_known"]], left_on="kr_session_date", right_on="date", how="left").drop(columns="date")
        else:
            daily["kr_sector_return_known"] = np.nan

        treasury_path = self._glob("us", "economic", "treasury", "US_TREASURY_daily")
        if list(treasury_path.rglob("*.parquet")):
            treasury = self._parquet(treasury_path).sort_values("date")
            treasury["control_available_utc"] = pd.to_datetime(treasury["date"], utc=True) + pd.Timedelta(days=1)
            treasury["us_2y_yield_change"] = pd.to_numeric(treasury["yield_2_year"], errors="coerce").diff()
            treasury["us_10y_yield_change"] = pd.to_numeric(treasury["yield_10_year"], errors="coerce").diff()
            daily = pd.merge_asof(daily.sort_values("us_close_ts_utc"), treasury[["control_available_utc", "us_2y_yield_change", "us_10y_yield_change"]], left_on="us_close_ts_utc", right_on="control_available_utc", direction="backward", tolerance=pd.Timedelta("7D"))

        inflation_path = self._glob("us", "economic", "inflation", "US_INFLATION_monthly")
        if list(inflation_path.rglob("*.parquet")):
            inflation = self._parquet(inflation_path)
            if "release_timestamp" in inflation:
                inflation["control_available_utc"] = pd.to_datetime(inflation["release_timestamp"], utc=True)
                inflation_column = "cpi_year_over_year" if "cpi_year_over_year" in inflation else "cpi_yoy"
                inflation["us_inflation_yoy"] = pd.to_numeric(inflation[inflation_column], errors="coerce")
                daily = pd.merge_asof(daily.sort_values("us_close_ts_utc"), inflation[["control_available_utc", "us_inflation_yoy"]].sort_values("control_available_utc"), left_on="us_close_ts_utc", right_on="control_available_utc", direction="backward", tolerance=pd.Timedelta("62D"), suffixes=(None, "_inflation"))

        proxy_path = self._glob("korea", "economic", "proxies", "KR_MARKET_PROXIES_daily")
        if list(proxy_path.rglob("*.parquet")):
            proxies = self._utc(self._parquet(proxy_path))
            proxies["proxy_return"] = proxies.groupby("ticker")["close"].transform(lambda values: np.log(values).diff())
            proxies["timestamp"] = proxies["timestamp"] + pd.Timedelta(days=1)
            for ticker, column in (("EWY", "korea_equity_risk_proxy"), ("C:USDKRW", "korea_currency_policy_expectations_proxy")):
                proxy = proxies.loc[proxies["ticker"].eq(ticker), ["timestamp", "proxy_return"]].rename(columns={"timestamp": f"{column}_ts", "proxy_return": column})
                daily = pd.merge_asof(daily.sort_values("us_close_ts_utc"), proxy.sort_values(f"{column}_ts"), left_on="us_close_ts_utc", right_on=f"{column}_ts", direction="backward", tolerance=pd.Timedelta("7D"))

        for column in {spec.source_column for spec in CONTROLS.values()}:
            if column not in daily:
                daily[column] = np.nan
        daily = add_parity_columns(daily, adr_column="adr_close_usd", kr_column="kr_close_krw", fx_column="usdkrw_asof")
        daily["adr_daily_return"] = np.log(daily["adr_close_usd"]).diff()
        daily["data_quality_flags"] = np.where(daily[["adr_close_usd", "kr_close_krw", "usdkrw_asof"]].isna().any(axis=1), "missing_required", "ok")
        return daily

    def _quote_buckets(self) -> pd.DataFrame:
        pattern = str(self._glob("us", "market", "SKHY_quotes") / "**" / "*.parquet").replace("\\", "/").replace("'", "''")
        return self.engine.query_df(f"""
          WITH normalized AS (
            SELECT CAST(sip_datetime AS TIMESTAMPTZ) AS quote_ts,
                   bid_price, ask_price, bid_size, ask_size
            FROM read_parquet('{pattern}')
            WHERE sip_datetime IS NOT NULL
              AND bid_price > 0 AND ask_price >= bid_price
          ),
          q AS (
            SELECT time_bucket(INTERVAL '5 minutes', quote_ts) AS bucket,
                   quote_ts, bid_price, ask_price, bid_size, ask_size
            FROM normalized
          )
          SELECT bucket, arg_max(bid_price, quote_ts) AS bid,
                 arg_max(ask_price, quote_ts) AS ask,
                 arg_max(bid_size, quote_ts) AS bid_size,
                 arg_max(ask_size, quote_ts) AS ask_size,
                 count(*) AS quote_updates,
                 count(DISTINCT date_trunc('minute', quote_ts))/5.0 AS quote_coverage,
                 max(quote_ts) AS last_quote_ts
          FROM q GROUP BY bucket ORDER BY bucket
        """)

    def build_us_5min(self) -> pd.DataFrame:
        bars = self._utc(self._parquet(self._glob("us", "market", "SKHY_5min")))
        bars = bars.rename(columns={"timestamp": "bar_start_utc", "transactions": "trade_count"})
        bars["bar_end_utc"] = bars["bar_start_utc"] + pd.Timedelta(minutes=5)
        local = bars["bar_start_utc"].dt.tz_convert(NY)
        bars["date_us"] = local.dt.date
        minute = local.dt.hour * 60 + local.dt.minute
        bars["session_type"] = np.select([minute < 570, minute < 960], ["premarket", "regular"], default="afterhours")
        bars["dollar_volume"] = bars["volume"] * bars.get("vwap", bars["close"])

        quotes = self._quote_buckets().rename(columns={"bucket": "bar_start_utc"})
        quotes["bar_start_utc"] = _utc_ns(quotes["bar_start_utc"])
        quotes["last_quote_ts"] = _utc_ns(quotes["last_quote_ts"])
        frame = bars.merge(quotes, on="bar_start_utc", how="left")
        frame["mid"] = (frame["bid"] + frame["ask"]) / 2
        frame["spread_bps"] = (frame["ask"] - frame["bid"]) / frame["mid"] * 10_000
        frame["quote_age_seconds"] = (frame["bar_end_utc"] - frame["last_quote_ts"]).dt.total_seconds()

        fx = self._utc(self._parquet(self._glob("fx", "USDKRW_1min")))[["timestamp", "close"]].rename(columns={"timestamp": "fx_ts", "close": "fx_mid_asof"})
        fx["fx_ts"] = fx["fx_ts"] + pd.Timedelta(minutes=1)
        frame = pd.merge_asof(frame.sort_values("bar_end_utc"), fx, left_on="bar_end_utc", right_on="fx_ts", direction="backward", tolerance=pd.Timedelta("15min"))
        kr = self._parquet(self._glob("korea", "market", "000660_daily"))
        kr["available"] = _utc_ns(
            pd.to_datetime(pd.to_datetime(kr["date"]).dt.date.astype(str) + " 15:30")
            .dt.tz_localize(SEOUL)
        )
        kr["kr_close_known"] = kr.get("adjusted_close", kr["close"])
        frame = pd.merge_asof(frame.sort_values("bar_end_utc"), kr[["available", "kr_close_known"]].sort_values("available"), left_on="bar_end_utc", right_on="available", direction="backward")
        frame = add_parity_columns(frame, adr_column="close", kr_column="kr_close_known", fx_column="fx_mid_asof")
        frame["valid_price"] = frame[["open", "high", "low", "close"]].gt(0).all(axis=1)
        frame["valid_quote"] = frame["mid"].gt(0) & frame["spread_bps"].ge(0)
        return frame

    @staticmethod
    def _attach_controls(frame: pd.DataFrame, controls: Iterable[str]) -> pd.DataFrame:
        output = frame.copy()
        for name in validate_controls(tuple(controls)):
            source = CONTROLS[name].source_column
            if source not in output:
                raise ValueError(f"Control {name!r} requires unavailable column {source!r}")
            if not output[source].notna().any():
                raise ValueError(f"Control {name!r} has no point-in-time observations in this sample")
            output[name] = output[source]
        return output

    def build_h1(self, max_lags: int = 5, controls: Iterable[str] = ("us_soxx_return", "kr_sector_return")) -> pd.DataFrame:
        # H1 variable construction lives in one place.  Import lazily to keep
        # the engine usable without loading the modelling stack at import time.
        from src.hypotheses.h1 import prepare_uecm_data

        control_names = tuple(controls)
        frame = self._attach_controls(self._require("daily"), control_names)
        frame = prepare_uecm_data(
            frame,
            max_p=max_lags,
            max_q=max_lags,
            controls=control_names,
        )
        required = ["a_d", "k_d", "delta_a", "delta_k", "a_l1", "k_l1"]
        frame["sample_eligible"] = frame[required].notna().all(axis=1)
        return frame

    @staticmethod
    def _asof_prices(
        events: pd.DataFrame,
        timestamp_column: str,
        prices: pd.DataFrame,
        output_column: str,
        *,
        tolerance: str = "60s",
    ) -> pd.DataFrame:
        left = events.sort_values(timestamp_column)
        right = prices.rename(
            columns={"timestamp": f"{output_column}_ts", "close": output_column}
        ).sort_values(f"{output_column}_ts")
        return pd.merge_asof(
            left,
            right,
            left_on=timestamp_column,
            right_on=f"{output_column}_ts",
            direction="backward",
            tolerance=pd.Timedelta(tolerance),
        ).sort_values("_row_id")

    def _minute_close_series(self, *parts: str) -> pd.DataFrame:
        frame = self._utc(self._parquet(self._glob(*parts)))[
            ["timestamp", "close"]
        ].copy()
        frame["timestamp"] = frame["timestamp"] + pd.Timedelta(minutes=1)
        frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
        return frame.loc[frame["close"].gt(0)].drop_duplicates(
            "timestamp", keep="last"
        )

    def _minute_open_series(self, *parts: str) -> pd.DataFrame:
        frame = self._utc(self._parquet(self._glob(*parts)))[
            ["timestamp", "open"]
        ].rename(columns={"open": "close"})
        frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
        return frame.loc[frame["close"].gt(0)].drop_duplicates(
            "timestamp", keep="last"
        )

    def build_h2(
        self,
        horizons: Iterable[int] = (30,),
        controls: Iterable[str] = ("us_soxx_return", "kr_sector_return"),
        h1_model_id: str = "theory_parity",
        h1_intercept: float | None = None,
        h1_theta: float | None = None,
        **_: object,
    ) -> pd.DataFrame:
        horizons = tuple(sorted(set(int(value) for value in horizons)))
        if horizons != (30,):
            raise ValueError("Primary H2 requires fixed 30-minute horizons")
        if h1_model_id != "theory_parity" and (
            h1_intercept is None or h1_theta is None
        ):
            raise ValueError("estimated H1 ECTs require fixed intercept and theta")
        base = pd.DataFrame({
            "date_us": pd.to_datetime(self._require("daily")["date_us"]).dt.date
        }).merge(pd.DataFrame({
            "anchor_type": [
                "fixed_0700", "fixed_0730", "fixed_0800",
                "fixed_0830", "fixed_0900", "regular_0930",
            ],
            "start_clock": ["07:00", "07:30", "08:00", "08:30", "09:00", "09:30"],
        }), how="cross")
        local_start = pd.to_datetime(
            base["date_us"].astype(str) + " " + base["start_clock"]
        ).dt.tz_localize(NY)
        base["anchor_ts_utc"] = local_start.dt.tz_convert("UTC")
        base["endpoint_ts_utc"] = base["anchor_ts_utc"] + pd.Timedelta(minutes=30)
        base["horizon_min"] = 30
        base["_row_id"] = np.arange(len(base))
        for name, parts in {
            "skhy": ("us", "market", "SKHY_1min"),
            "soxx": ("us", "market", "SOXX_1min"),
            "usdkrw": ("fx", "USDKRW_1min"),
        }.items():
            base = self._asof_prices(
                base, "anchor_ts_utc", self._minute_open_series(*parts),
                f"{name}_anchor_price",
            )
            base = self._asof_prices(
                base, "endpoint_ts_utc", self._minute_close_series(*parts),
                f"{name}_horizon",
            )
        kr = self._parquet(self._glob("korea", "market", "000660_daily")).copy()
        kr["available_at_utc"] = _utc_ns(
            pd.to_datetime(pd.to_datetime(kr["date"]).dt.date.astype(str) + " 15:30")
            .dt.tz_localize(SEOUL)
        )
        kr_price = pd.to_numeric(kr.get("adjusted_close", kr["close"]), errors="coerce")
        kr["kr_close_known"] = kr_price
        kr["kr_return_known"] = np.log(kr_price).diff()
        base = pd.merge_asof(
            base.sort_values("anchor_ts_utc"),
            kr[["available_at_utc", "kr_close_known", "kr_return_known"]].sort_values("available_at_utc"),
            left_on="anchor_ts_utc", right_on="available_at_utc", direction="backward",
        )
        intercept = 0.0 if h1_intercept is None else h1_intercept
        theta = 1.0 if h1_theta is None else h1_theta
        base["ect_anchor"] = (
            np.log(base["skhy_anchor_price"])
            - intercept
            - theta * np.log(base["kr_close_known"] * 0.1 / base["usdkrw_anchor_price"])
        )
        base["h1_model_id"] = h1_model_id
        base["online_eligible"] = True
        expanded = base.copy()
        expanded["outcome_ts"] = expanded["skhy_horizon_ts"]
        expanded["anchor_price"] = expanded["skhy_anchor_price"]
        expanded["outcome_price"] = expanded["skhy_horizon"]
        expanded["adr_return_h"] = np.log(
            expanded["skhy_horizon"] / expanded["skhy_anchor_price"]
        )
        expanded["soxx_return_h"] = np.log(
            expanded["soxx_horizon"] / expanded["soxx_anchor_price"]
        )
        expanded["usdkrw_return_h"] = np.log(
            expanded["usdkrw_horizon"] / expanded["usdkrw_anchor_price"]
        )
        expanded["anchor_valid"] = expanded[[
            "skhy_anchor_price", "soxx_anchor_price", "usdkrw_anchor_price",
        ]].notna().all(axis=1)
        expanded["selection_status"] = np.where(
            expanded["anchor_valid"], "fixed_window", "fixed_window_unavailable"
        )
        expanded["anchor_mid_usd"] = expanded["skhy_anchor_price"]
        expanded["anchor_trade_usd"] = expanded["skhy_anchor_price"]
        expanded["liquidity_score"] = np.nan
        expanded["spread_bps"] = np.nan
        expanded["volume_5min"] = np.nan
        expanded["trade_count_5min"] = np.nan
        expanded["quote_updates_5min"] = np.nan
        expanded["usdkrw_anchor"] = expanded["usdkrw_anchor_price"]
        expanded["parity_spread_anchor"] = expanded["ect_anchor"]
        expanded["selection_version"] = "fixed-30m-v1"
        expanded["ect_x_liquidity"] = np.nan
        expanded["ect_change_h"] = expanded["adr_return_h"]
        open_ts = pd.to_datetime(
            expanded["date_us"].astype(str) + " 09:30"
        ).dt.tz_localize(NY).dt.tz_convert("UTC")
        expanded["crosses_open"] = (
            expanded["anchor_ts_utc"] < open_ts
        ) & (expanded["endpoint_ts_utc"] > open_ts)
        expanded["valid_outcome"] = (
            expanded["online_eligible"].fillna(False)
            & expanded[[
                "ect_anchor", "skhy_anchor_price", "skhy_horizon",
                "soxx_anchor_price", "soxx_horizon", "usdkrw_anchor_price",
                "usdkrw_horizon", "kr_return_known",
            ]].notna().all(axis=1)
            & expanded[[
                "skhy_anchor_price", "skhy_horizon", "soxx_anchor_price",
                "soxx_horizon", "usdkrw_anchor_price", "usdkrw_horizon",
            ]].gt(0).all(axis=1)
        )

        # Preserve registered daily controls for H3 and robustness models. The
        # primary pooled H2 ECM does not use these columns.
        control_daily = self._require("daily").sort_values("date_us").copy()
        control_daily["us_sector_return"] = control_daily["us_sector_return"].shift(1)
        daily = self._attach_controls(control_daily, controls)
        control_columns = ["date_us", *validate_controls(tuple(controls))]
        expanded = expanded.merge(daily[control_columns], on="date_us", how="left")
        return expanded.drop(columns=["_row_id", "start_clock"])

    def build_h3(self, h1_model_id: str = "theory_parity", h1_intercept: float | None = None, h1_theta: float | None = None, controls: Iterable[str] = ("us_soxx_return", "kr_sector_return")) -> pd.DataFrame:
        if h1_model_id != "theory_parity" and (h1_intercept is None or h1_theta is None):
            raise ValueError("estimated H1 ECTs require h1_intercept and h1_theta")
        daily = self._attach_controls(self._require("daily"), controls).sort_values("date_us")
        daily = daily.copy()
        daily["us_close_ts_utc"] = _utc_ns(daily["us_close_ts_utc"])
        intercept = 0.0 if h1_intercept is None else h1_intercept
        theta = 1.0 if h1_theta is None else h1_theta
        daily["ect_pre_response"] = (
            daily["log_adr"] - intercept - theta * daily["log_kr_usd"]
        )
        kr = self._parquet(self._glob("korea", "market", "000660_daily"))
        kr["target_session_date"] = pd.to_datetime(kr["date"]).dt.date
        kr["target_available_utc"] = _utc_ns(
            pd.to_datetime(kr["target_session_date"].astype(str) + " 15:30")
            .dt.tz_localize(SEOUL)
        )
        kr_price = kr.get("adjusted_close", kr["close"])
        kr["target_market_return"] = np.log(kr_price).diff()
        paired = pd.merge_asof(
            daily.sort_values("us_close_ts_utc"),
            kr[[
                "target_session_date", "target_available_utc",
                "target_market_return",
            ]].sort_values("target_available_utc"),
            left_on="us_close_ts_utc",
            right_on="target_available_utc",
            direction="forward",
            allow_exact_matches=False,
        )
        us_to_korea = pd.DataFrame({
            "direction": "us_to_korea", "source_session_date": paired["date_us"], "target_session_date": paired["target_session_date"],
            "information_cutoff_utc": paired["us_close_ts_utc"], "anchor_type": "us_regular_close", "horizon_min": 0,
            "ect_pre_response": paired["ect_pre_response"], "source_market_return": paired["adr_daily_return"], "target_market_return": paired["target_market_return"],
            "valid_outcome": paired["target_market_return"].notna() & paired["ect_pre_response"].notna(), "h1_model_id": h1_model_id,
        }).dropna(subset=["source_session_date", "target_session_date", "information_cutoff_utc"])
        us_to_korea = us_to_korea.sort_values("source_session_date").drop_duplicates(
            "target_session_date", keep="last"
        )
        for control in validate_controls(tuple(controls)):
            us_to_korea[control] = paired[control]
        return us_to_korea.reset_index(drop=True)

    def _require(self, name: str) -> pd.DataFrame:
        if name not in self.frames:
            self.frames[name] = getattr(self, FEATURES[name].builder)()
        return self.frames[name]

    def build(self, targets: str | Iterable[str], *, experiment: ExperimentSpec | None = None, run_id: str | None = None, materialize: bool = True, force: bool = False) -> dict[str, DatasetBuildResult]:
        requested = (targets,) if isinstance(targets, str) else tuple(targets)
        if experiment:
            experiment.validate()
            materialize = materialize and experiment.materialize_features
        results: dict[str, DatasetBuildResult] = {}
        for name in resolve_dependencies(requested):
            source_digest = snapshot_hash(self._raw_source_paths(name))
            if experiment is None or name in {"daily", "us_5min"}:
                experiment_digest = "default"
            elif name == "h1":
                relevant = {"max_lags": experiment.max_lags, "controls": experiment.controls, "start": str(experiment.start_date), "end": str(experiment.end_date)}
                experiment_digest = hashlib.sha256(json.dumps(relevant, sort_keys=True).encode()).hexdigest()
            else:
                experiment_digest = experiment.fingerprint
            cache_key = self.cache.key(name, FEATURES[name].version, source_digest, experiment_digest)
            model_run = (
                (run_id or experiment.fingerprint)
                if experiment and name in {"h1", "h2", "h3"}
                else None
            )
            cache_slot = f"{name}:{model_run}" if model_run else name
            output = self.cache.output_path(name, model_run)
            if materialize and not force and self.cache.is_valid(cache_slot, cache_key):
                frame = _normalize_contract_timestamps(
                    pd.read_parquet(output), name
                )
                self.frames[name] = frame
                print(f"[build] {name}: cache hit ({len(frame):,} rows)", flush=True)
                results[name] = DatasetBuildResult(name, frame, output, True)
                continue
            kwargs: dict[str, object] = {}
            if experiment and name == "h1": kwargs.update(max_lags=experiment.max_lags, controls=experiment.controls)
            if experiment and name == "h2": kwargs.update(
                horizons=experiment.horizons,
                controls=experiment.controls,
                h1_model_id=experiment.h1_model_id,
                h1_intercept=experiment.h1_intercept,
                h1_theta=experiment.h1_theta,
            )
            if experiment and name == "h3": kwargs.update(h1_model_id=experiment.h1_model_id, h1_intercept=experiment.h1_intercept, h1_theta=experiment.h1_theta, controls=experiment.controls)
            print(f"[build] {name}: starting", flush=True)
            frame = getattr(self, FEATURES[name].builder)(**kwargs)
            if experiment and name in {"h1", "h2", "h3"}:
                date_column = (
                    "target_session_date" if name == "h3"
                    else "date_us"
                )
                dates = pd.to_datetime(frame[date_column]).dt.date
                if experiment.start_date:
                    frame = frame.loc[dates >= experiment.start_date].copy()
                    dates = pd.to_datetime(frame[date_column]).dt.date
                if experiment.end_date:
                    frame = frame.loc[dates <= experiment.end_date].copy()
            validate_contract(frame, name)
            if name in {"daily", "us_5min", "h2", "h3"}:
                validate_no_lookahead(frame, name)
            self.frames[name] = frame
            output = None
            if materialize:
                output = self.cache.output_path(name, model_run)
                self.engine.write_parquet(frame, output)
                self.cache.record(cache_slot, cache_key, output, len(frame))
            print(f"[build] {name}: {len(frame):,} rows" + (f" -> {output}" if output else ""), flush=True)
            results[name] = DatasetBuildResult(name, frame, output)
        return {name: results[name] for name in requested}
