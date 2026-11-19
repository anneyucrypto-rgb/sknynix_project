from datetime import datetime, timezone

from src.ingestion.massive_news import aggregate_sentiment, normalize_ticker_sentiment


def test_ticker_sentiment_mapping_and_minute_aggregation():
    articles = [
        {
            "id": "a",
            "published_utc": "2026-01-02T12:01:10Z",
            "title": "Positive",
            "insights": [{"ticker": "SKHY", "sentiment": "positive"}],
        },
        {
            "id": "b",
            "published_utc": "2026-01-02T12:04:20Z",
            "title": "Negative",
            "insights": [{"ticker": "SKHY", "sentiment": "negative"}],
        },
        {
            "id": "c",
            "published_utc": "2026-01-02T12:04:40Z",
            "title": "Other ticker",
            "insights": [{"ticker": "OTHER", "sentiment": "positive"}],
        },
    ]
    events = normalize_ticker_sentiment(
        articles, "SKHY", received_at=datetime(2026, 1, 2, tzinfo=timezone.utc)
    )
    assert events["sentiment_score"].tolist() == [1, -1]
    one_minute = aggregate_sentiment(events, 1)
    assert one_minute["sentiment_score"].tolist() == [1.0, -1.0]
    five_minute = aggregate_sentiment(events, 5).iloc[0]
    assert five_minute["article_count"] == 2
    assert five_minute["sentiment_sum"] == 0
    assert five_minute["sentiment_score"] == 0.0
    assert five_minute["sentiment_log_odds"] == 0.0
    assert five_minute["sentiment_log_odds_normalized"] == 0.0


def test_neutral_only_bucket_normalizes_to_zero():
    articles = [{
        "id": "neutral",
        "published_utc": "2026-01-02T12:01:10Z",
        "insights": [{"ticker": "SKHY", "sentiment": "neutral"}],
    }]
    events = normalize_ticker_sentiment(articles, "SKHY")
    bucket = aggregate_sentiment(events, 1).iloc[0]
    assert bucket["sentiment_score"] == 0.0
    assert bucket["sentiment_log_odds_normalized"] == 0.0
