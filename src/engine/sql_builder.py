"""Strict SQL template loading, rendering, and hashing."""

from __future__ import annotations

import hashlib
from pathlib import Path
import re
from typing import Mapping

from src.config import PROJECT_ROOT
from src.engine.duckdb_engine import quote_identifier


_TOKEN = re.compile(r"{{\s*([A-Za-z_][A-Za-z0-9_]*)\s*}}")


def sql_literal(value: object) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, float)):
        return str(value)
    return "'" + str(value).replace("'", "''") + "'"


class SQLBuilder:
    def __init__(self, root: str | Path = PROJECT_ROOT) -> None:
        self.root = Path(root).resolve()

    def load(self, relative_path: str | Path) -> str:
        path = (self.root / relative_path).resolve()
        if self.root not in path.parents:
            raise ValueError("SQL path escapes the project root")
        return path.read_text(encoding="utf-8")

    def render(self, template: str, values: Mapping[str, object]) -> str:
        needed = set(_TOKEN.findall(template))
        missing = needed.difference(values)
        extra = set(values).difference(needed)
        if missing:
            raise KeyError(f"Missing SQL parameters: {sorted(missing)}")
        if extra:
            raise KeyError(f"Unused SQL parameters: {sorted(extra)}")
        return _TOKEN.sub(lambda match: str(values[match.group(1)]), template)

    def compile(self, relative_path: str | Path, values: Mapping[str, object]) -> str:
        return self.render(self.load(relative_path), values)

    @staticmethod
    def digest(sql: str) -> str:
        normalized = " ".join(sql.split())
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


__all__ = ["SQLBuilder", "quote_identifier", "sql_literal"]
