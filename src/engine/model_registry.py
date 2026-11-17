"""DuckDB-backed experiment, coefficient, and diagnostic registry."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from typing import Mapping

import pandas as pd

from src.engine.duckdb_engine import DuckDBEngine
from src.engine.experiment import ExperimentSpec


class ModelRegistry:
    def __init__(self, engine: DuckDBEngine) -> None:
        self.engine = engine
        self.engine.execute("""
          CREATE TABLE IF NOT EXISTS experiment_runs (
            run_id VARCHAR PRIMARY KEY, hypothesis_id VARCHAR NOT NULL,
            estimator VARCHAR NOT NULL, status VARCHAR NOT NULL,
            spec_hash VARCHAR NOT NULL, spec_json JSON NOT NULL,
            source_snapshot_hash VARCHAR, sql_hash VARCHAR,
            feature_version VARCHAR, h1_model_id VARCHAR,
            training_start DATE, training_end DATE,
            evaluation_start DATE, evaluation_end DATE,
            effective_n BIGINT, estimator_version VARCHAR,
            diagnostic_status VARCHAR, report_path VARCHAR,
            created_at_utc TIMESTAMPTZ NOT NULL,
            finished_at_utc TIMESTAMPTZ, message VARCHAR
          )
        """)
        self.engine.execute("""
          CREATE TABLE IF NOT EXISTS model_coefficients (
            run_id VARCHAR, term VARCHAR, estimate DOUBLE, std_error DOUBLE,
            statistic DOUBLE, p_value DOUBLE, conf_low DOUBLE, conf_high DOUBLE,
            status VARCHAR, PRIMARY KEY(run_id, term)
          )
        """)
        self.engine.execute("""
          CREATE TABLE IF NOT EXISTS model_statistics (
            run_id VARCHAR, name VARCHAR, value DOUBLE, p_value DOUBLE,
            status VARCHAR, detail VARCHAR,
            PRIMARY KEY(run_id, name)
          )
        """)

    def start(
        self, spec: ExperimentSpec, run_id: str | None = None, *,
        source_snapshot_hash: str | None = None, sql_hash: str | None = None,
        feature_version: str | None = None, estimator_version: str | None = None,
    ) -> str:
        identifier = run_id or spec.new_run_id()
        self.engine.execute(
            """INSERT INTO experiment_runs (
              run_id, hypothesis_id, estimator, status, spec_hash, spec_json,
              source_snapshot_hash, sql_hash, feature_version, h1_model_id,
              evaluation_start, evaluation_end, estimator_version, created_at_utc
            ) VALUES (?, ?, ?, 'running', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [identifier, spec.hypothesis.upper(), spec.estimator, spec.fingerprint,
             spec.canonical_json(), source_snapshot_hash, sql_hash, feature_version,
             spec.h1_model_id, spec.start_date, spec.end_date, estimator_version,
             datetime.now(timezone.utc)],
        )
        self.engine.execute(
            "UPDATE experiment_runs SET training_start=?, training_end=? WHERE run_id=?",
            [spec.training_start_date, spec.training_end_date, identifier],
        )
        return identifier

    def finish(
        self, run_id: str, *, status: str = "complete", message: str | None = None,
        effective_n: int | None = None, diagnostic_status: str | None = None,
        report_path: str | None = None,
    ) -> None:
        if status not in {"complete", "failed", "cancelled"}:
            raise ValueError("invalid terminal status")
        self.engine.execute(
            """UPDATE experiment_runs SET status=?, finished_at_utc=?, message=?,
                 effective_n=?, diagnostic_status=?, report_path=? WHERE run_id=?""",
            [status, datetime.now(timezone.utc), message, effective_n,
             diagnostic_status, report_path, run_id],
        )

    def save_coefficients(self, run_id: str, coefficients: pd.DataFrame) -> None:
        required = {"term", "estimate"}
        if missing := required.difference(coefficients.columns):
            raise ValueError(f"Missing coefficient columns: {sorted(missing)}")
        for row in coefficients.to_dict("records"):
            self.engine.execute("""
              INSERT OR REPLACE INTO model_coefficients VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [run_id, row["term"], row["estimate"], row.get("std_error"),
                   row.get("statistic"), row.get("p_value"), row.get("conf_low"),
                   row.get("conf_high"), row.get("status")])

    def save_statistics(self, run_id: str, values: Mapping[str, float], details: Mapping[str, object] | None = None) -> None:
        for name, value in values.items():
            payload = (details or {}).get(name)
            detail = json.dumps(payload) if payload is not None else None
            p_value = payload.get("p_value") if isinstance(payload, dict) else None
            status = payload.get("status") if isinstance(payload, dict) else None
            self.engine.execute("INSERT OR REPLACE INTO model_statistics VALUES (?, ?, ?, ?, ?, ?)", [run_id, name, value, p_value, status, detail])

    def runs(self) -> pd.DataFrame:
        return self.engine.query_df("SELECT * FROM experiment_runs ORDER BY started_at_utc DESC")
