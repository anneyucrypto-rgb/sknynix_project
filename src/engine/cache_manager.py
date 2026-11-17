"""Content-addressed feature cache metadata."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from functools import lru_cache
from pathlib import Path
import re
from typing import Iterable

from src.config import PathSettings
from src.engine.duckdb_engine import DuckDBEngine


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@lru_cache(maxsize=4096)
def _versioned_file_hash(path: str, size: int, modified_ns: int) -> str:
    """Hash an immutable snapshot once per process; stat fields invalidate it."""

    return file_sha256(path)


def snapshot_hash(paths: Iterable[str | Path]) -> str:
    digest = hashlib.sha256()
    files: list[Path] = []
    for candidate in paths:
        path = Path(candidate)
        files.extend(path.rglob("*.parquet") if path.is_dir() else [path])
    for path in sorted(files, key=lambda item: str(item)):
        if path.is_file():
            digest.update(str(path).encode())
            stat = path.stat()
            digest.update(str(stat.st_size).encode())
            digest.update(_versioned_file_hash(str(path), stat.st_size, stat.st_mtime_ns).encode())
    return digest.hexdigest()


class CacheManager:
    def __init__(self, engine: DuckDBEngine, paths: PathSettings | None = None) -> None:
        self.engine = engine
        self.paths = paths or PathSettings()
        self.engine.execute("""
            CREATE TABLE IF NOT EXISTS feature_cache (
                dataset VARCHAR PRIMARY KEY, cache_key VARCHAR NOT NULL,
                output_path VARCHAR NOT NULL, row_count BIGINT,
                created_at_utc TIMESTAMPTZ NOT NULL
            )
        """)

    @staticmethod
    def key(*parts: str) -> str:
        return hashlib.sha256("\0".join(parts).encode()).hexdigest()

    def output_path(self, dataset: str, run_id: str | None = None) -> Path:
        if run_id is not None:
            if not re.fullmatch(r"[A-Za-z0-9._=-]+", run_id):
                raise ValueError("run_id contains unsafe path characters")
            return self.paths.data / "features" / "experiments" / f"run_id={run_id}" / f"{dataset}.parquet"
        return self.paths.data / "features" / dataset / "part-00000.parquet"

    def is_valid(self, dataset: str, cache_key: str) -> bool:
        row = self.engine.execute(
            "SELECT cache_key, output_path FROM feature_cache WHERE dataset = ?", [dataset]
        ).fetchone()
        return bool(row and row[0] == cache_key and Path(row[1]).is_file())

    def record(self, dataset: str, cache_key: str, output_path: Path, row_count: int) -> None:
        self.engine.execute("""
            INSERT INTO feature_cache VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(dataset) DO UPDATE SET cache_key=excluded.cache_key,
              output_path=excluded.output_path, row_count=excluded.row_count,
              created_at_utc=excluded.created_at_utc
        """, [dataset, cache_key, str(output_path), row_count, datetime.now(timezone.utc)])

    def invalidate(self, datasets: Iterable[str]) -> None:
        for dataset in datasets:
            self.engine.execute("DELETE FROM feature_cache WHERE dataset = ?", [dataset])
