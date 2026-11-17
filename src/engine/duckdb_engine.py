"""Small, safe DuckDB wrapper used by builders and registries."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import re
from typing import Iterator, Mapping, Sequence

import duckdb
import pandas as pd

from src.config import PathSettings


_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def quote_identifier(value: str) -> str:
    if not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"Unsafe SQL identifier: {value!r}")
    return f'"{value}"'


class DuckDBEngine:
    def __init__(self, path: str | Path | None = None, *, read_only: bool = False) -> None:
        self.path = Path(path) if path else PathSettings().duckdb
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = duckdb.connect(str(self.path), read_only=read_only)
        self.connection.execute("SET TimeZone = 'UTC'")

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "DuckDBEngine":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def execute(self, sql: str, parameters: Sequence[object] | Mapping[str, object] | None = None):
        return self.connection.execute(sql, parameters) if parameters is not None else self.connection.execute(sql)

    def query_df(self, sql: str, parameters: Sequence[object] | None = None) -> pd.DataFrame:
        cursor = self.connection.execute(sql, parameters) if parameters is not None else self.connection.execute(sql)
        return cursor.fetchdf()

    def register_dataframe(self, name: str, frame: pd.DataFrame) -> None:
        quote_identifier(name)
        self.connection.register(name, frame)

    def register_parquet_view(self, name: str, path: str | Path) -> None:
        identifier = quote_identifier(name)
        pattern = str(Path(path)).replace("\\", "/").replace("'", "''")
        self.execute(f"CREATE OR REPLACE VIEW {identifier} AS SELECT * FROM read_parquet('{pattern}')")

    def relation_exists(self, name: str) -> bool:
        return bool(self.execute(
            "SELECT count(*) FROM information_schema.tables WHERE table_name = ?",
            [name],
        ).fetchone()[0])

    @contextmanager
    def transaction(self) -> Iterator[None]:
        self.execute("BEGIN")
        try:
            yield
        except Exception:
            self.execute("ROLLBACK")
            raise
        else:
            self.execute("COMMIT")

    def write_parquet(self, frame: pd.DataFrame, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        if temporary.exists():
            temporary.unlink()
        self.register_dataframe("_materialize_frame", frame)
        escaped = str(temporary).replace("\\", "/").replace("'", "''")
        self.execute(f"COPY _materialize_frame TO '{escaped}' (FORMAT PARQUET, COMPRESSION ZSTD)")
        temporary.replace(destination)
        return destination
