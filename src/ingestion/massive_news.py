"""Download ticker-specific Massive LLM news sentiment and minute buckets."""

from __future__ import annotations

import argparse
import json
import os
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np

from src.config import IngestionSettings, read_secret
from src.ingestion.massive_us import (
    DateRange,
    MassiveClient,
    merge_tabular_snapshot,
    print_page,
    print_status,
    print_task,
    write_parquet,
)


SENTIMENT_SCORES = {"positive": 1, "neutral": 0, "negative": -1}
EVENT_COLUMNS = (
    "article_id",
    "ticker",
    "published_utc",
    "sentiment_label",
    "sentiment_score",
    "sentiment_reasoning",
    "publisher_name",
    "title",
    "description",
    "author",
    "article_url",
    "article_tickers",
    "keywords",
    "received_at_utc",
)


def _json_array(value: Any) -> str:
    return json.dumps(value if isinstance(value, list) else [], separators=(",", ":"))


def normalize_ticker_sentiment(
    articles: list[dict[str, Any]], ticker: str, *, received_at: datetime | None = None
) -> pd.DataFrame:
    """Extract only the requested ticker's scored insight from each article."""

    ticker = ticker.upper()
    received = received_at or datetime.now(timezone.utc)
    rows: list[dict[str, Any]] = []
    for article in articles:
        matching = next(
            (
                insight
                for insight in article.get("insights") or []
                if str(insight.get("ticker", "")).upper() == ticker
            ),
            None,
        )
        if matching is None:
            continue
        label = str(matching.get("sentiment", "")).lower()
        if label not in SENTIMENT_SCORES:
            continue
        published = pd.to_datetime(article.get("published_utc"), utc=True, errors="coerce")
        if pd.isna(published):
            continue
        publisher = article.get("publisher") or {}
        rows.append(
            {
                "article_id": str(article.get("id", "")),
                "ticker": ticker,
                "published_utc": published,
                "sentiment_label": label,
                "sentiment_score": SENTIMENT_SCORES[label],
                "sentiment_reasoning": matching.get("sentiment_reasoning"),
                "publisher_name": publisher.get("name"),
                "title": article.get("title"),
                "description": article.get("description"),
                "author": article.get("author"),
                "article_url": article.get("article_url"),
                "article_tickers": _json_array(article.get("tickers")),
                "keywords": _json_array(article.get("keywords")),
                "received_at_utc": pd.Timestamp(received),
            }
        )
    return pd.DataFrame(rows, columns=EVENT_COLUMNS)


def aggregate_sentiment(events: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """Aggregate mapped article sentiment into sparse UTC minute buckets."""

    columns = [
        "ticker", "timestamp", "interval_minutes", "article_count",
        "positive_count", "neutral_count", "negative_count",
        "sentiment_sum", "sentiment_score", "sentiment_log_odds",
        "sentiment_log_odds_normalized", "latest_published_utc", "article_ids",
    ]
    if events.empty:
        return pd.DataFrame(columns=columns)
    frame = events.copy()
    frame["published_utc"] = pd.to_datetime(frame["published_utc"], utc=True)
    frame["timestamp"] = frame["published_utc"].dt.floor(f"{minutes}min")
    frame["positive_count"] = frame["sentiment_label"].eq("positive").astype(int)
    frame["neutral_count"] = frame["sentiment_label"].eq("neutral").astype(int)
    frame["negative_count"] = frame["sentiment_label"].eq("negative").astype(int)
    grouped = frame.groupby(["ticker", "timestamp"], as_index=False, sort=True)
    output = grouped.agg(
        article_count=("article_id", "size"),
        positive_count=("positive_count", "sum"),
        neutral_count=("neutral_count", "sum"),
        negative_count=("negative_count", "sum"),
        sentiment_sum=("sentiment_score", "sum"),
        sentiment_score=("sentiment_score", "mean"),
        latest_published_utc=("published_utc", "max"),
        article_ids=("article_id", lambda values: json.dumps(list(values), separators=(",", ":"))),
    )
    # Primary score: bounded net article balance. This is equivalent to the
    # mean of the article-level {-1, 0, +1} scores and lets neutral articles
    # shrink the signal toward zero.
    output["sentiment_score"] = np.where(
        output["article_count"].gt(0),
        output["sentiment_sum"] / output["article_count"],
        0.0,
    )
    # Secondary score: Jeffreys-smoothed positive/negative log odds. The
    # bounded transform is easier to compare across buckets and equals zero
    # when positive and negative counts are both zero.
    output["sentiment_log_odds"] = np.log(
        (output["positive_count"] + 0.5) / (output["negative_count"] + 0.5)
    )
    output["sentiment_log_odds_normalized"] = np.tanh(
        output["sentiment_log_odds"] / 2.0
    )
    output.insert(2, "interval_minutes", minutes)
    return output[columns]


def _coverage_path(events_path: Path) -> Path:
    return events_path.with_name("coverage.json")


def _coverage_gaps(path: Path, start: date, end: date, force: bool) -> tuple[DateRange, ...]:
    if force or not path.is_file():
        return (DateRange(start, end),)
    try:
        coverage = json.loads(path.read_text(encoding="utf-8"))
        covered_start = date.fromisoformat(coverage["start_date"])
        covered_end = date.fromisoformat(coverage["end_date"])
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return (DateRange(start, end),)
    gaps: list[DateRange] = []
    if start < covered_start:
        gaps.append(DateRange(start, min(end, covered_start - timedelta(days=1))))
    if end > covered_end:
        gaps.append(DateRange(max(start, covered_end + timedelta(days=1)), end))
    return tuple(gap for gap in gaps if gap.start <= gap.end)


def _write_coverage(path: Path, start: date, end: date, force: bool) -> None:
    if path.is_file() and not force:
        try:
            current = json.loads(path.read_text(encoding="utf-8"))
            start = min(start, date.fromisoformat(current["start_date"]))
            end = max(end, date.fromisoformat(current["end_date"]))
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            pass
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.part")
    temporary.write_text(
        json.dumps({"start_date": start.isoformat(), "end_date": end.isoformat()}, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def download_news_sentiment(
    start: date,
    end: date,
    *,
    ticker: str | None = None,
    force: bool = False,
) -> dict[str, int | str]:
    """Download and materialize ticker sentiment events plus 1/5-minute bins."""

    if start > end:
        raise ValueError("start must not be later than end")
    settings = IngestionSettings()
    ticker = (ticker or settings.massive.ticker).upper()
    client = MassiveClient(
        read_secret(settings.massive.api_key_env),
        base_url=settings.massive.base_url,
        timeout=settings.massive.timeout_seconds,
        max_retries=settings.massive.max_retries,
    )
    events_path = settings.paths.raw_part("us", "news", f"{ticker}_sentiment_events")
    coverage_path = _coverage_path(events_path)
    gaps = _coverage_gaps(coverage_path, start, end, force)
    results: dict[str, int | str] = {}
    print_status(f"{ticker} news sentiment ingestion started for {start} through {end}")
    additions: list[pd.DataFrame] = []
    received_at = datetime.now(timezone.utc)
    if gaps:
        print_task(1, 3, f"{ticker} scored news events", "DOWNLOADING " + ", ".join(g.label() for g in gaps))
        for gap_index, gap in enumerate(gaps, start=1):
            articles: list[dict[str, Any]] = []
            params = {
                "ticker": ticker,
                "published_utc.gte": datetime.combine(gap.start, time.min, timezone.utc).isoformat().replace("+00:00", "Z"),
                "published_utc.lte": datetime.combine(gap.end, time.max, timezone.utc).isoformat().replace("+00:00", "Z"),
                "sort": "published_utc",
                "order": "asc",
                "limit": 1000,
            }
            for page_number, page in enumerate(client.iter_pages("/v2/reference/news", params), start=1):
                articles.extend(page)
                print_page(f"{ticker} news gap {gap_index}/{len(gaps)}", page_number, len(articles))
            additions.append(normalize_ticker_sentiment(articles, ticker, received_at=received_at))
        if force:
            events = pd.concat(additions, ignore_index=True) if additions else pd.DataFrame(columns=EVENT_COLUMNS)
            events = events.drop_duplicates(["article_id", "ticker"], keep="last").sort_values(["published_utc", "article_id"])
            write_parquet(events, events_path)
        else:
            merge_tabular_snapshot(events_path, additions, keys=("article_id", "ticker"))
        _write_coverage(coverage_path, start, end, force)
    else:
        print_task(1, 3, f"{ticker} scored news events", "SKIPPED")

    if events_path.is_file():
        events = pd.read_parquet(events_path).sort_values(
            ["published_utc", "article_id"]
        ).reset_index(drop=True)
        write_parquet(events, events_path)
    else:
        events = pd.DataFrame(columns=EVENT_COLUMNS)
        write_parquet(events, events_path)
    results[str(events_path)] = len(events)
    for task_index, minutes in enumerate((1, 5), start=2):
        destination = settings.paths.raw_part("us", "news", f"{ticker}_sentiment_{minutes}min")
        buckets = aggregate_sentiment(events, minutes)
        write_parquet(buckets, destination)
        results[str(destination)] = len(buckets)
        print_task(task_index, 3, f"{ticker} {minutes}-minute sentiment", f"DONE {len(buckets):,} rows")
    print_status(f"{ticker} news sentiment ingestion complete")
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", type=date.fromisoformat, default=date(2024, 7, 2))
    parser.add_argument("--end-date", type=date.fromisoformat, default=date.today())
    parser.add_argument("--ticker", default=None, help="Massive U.S. ticker (default: configured SKHY)")
    parser.add_argument("--force", "--refresh-all", dest="force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results = download_news_sentiment(args.start_date, args.end_date, ticker=args.ticker, force=args.force)
    for path, rows in results.items():
        print(f"{rows:>12,} rows -> {path}" if isinstance(rows, int) else f"{rows:>16} -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
