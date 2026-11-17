import pandas as pd

from src.features.anchor_selector import AnchorSpec, select_anchors


def test_dynamic_anchor_requires_two_completed_qualifying_bars():
    starts = pd.to_datetime(["2026-01-05 12:00Z", "2026-01-05 12:05Z", "2026-01-05 12:10Z", "2026-01-05 14:30Z"])
    frame = pd.DataFrame({
        "date_us": [pd.Timestamp("2026-01-05").date()] * 4,
        "bar_start_utc": starts, "bar_end_utc": starts + pd.Timedelta("5min"),
        "open": [10] * 4, "close": [10] * 4, "mid": [10] * 4,
        "volume": [2000] * 4, "trade_count": [20] * 4, "quote_updates": [20] * 4,
        "spread_bps": [10] * 4, "quote_coverage": [1.0] * 4, "quote_age_seconds": [1] * 4,
        "valid_price": [True] * 4, "valid_quote": [True] * 4,
        "kr_close_known": [100000] * 4, "fx_mid_asof": [1300] * 4, "parity_spread": [0.1] * 4,
    })
    anchors = select_anchors(frame, AnchorSpec(min_liquidity_score=-100))
    dynamic = anchors.query("anchor_type == 'premarket_dynamic'").iloc[0]
    assert dynamic.anchor_ts_utc == pd.Timestamp("2026-01-05 12:10Z")
    assert anchors.anchor_valid.all()
