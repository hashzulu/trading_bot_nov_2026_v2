"""Tests for the pivot + prominence + isolation real-swing detector."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from botv2.analysis import build_view
from botv2.config import Config
from botv2.features import (Structure, Swing, atr, bos_events, bos_reference,
                            choch_events, detect_ranges, range_breakout_events,
                            real_swings, resolve_range_event, rolling_atr,
                            sweep_reversal_events)


def _frame(n: int = 60) -> pd.DataFrame:
    index = pd.DatetimeIndex(
        [datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=i) for i in range(n)]
    )
    close = np.full(n, 100.0)
    high = np.full(n, 100.0)
    low = np.full(n, 99.6)
    return pd.DataFrame(
        {"open": close, "high": high, "low": low, "close": close, "volume": np.ones(n)},
        index=index,
    )


def _scan(df: pd.DataFrame, **kwargs):
    return real_swings(df, atr(df, 14), **kwargs)


def test_clear_high_and_low_are_found_with_confirmation():
    df = _frame()
    df.loc[df.index[25], ["high", "low", "close"]] = [106.0, 100.0, 101.0]
    df.loc[df.index[45], ["high", "low", "close"]] = [100.0, 94.0, 99.0]

    swings = _scan(df, n=2, prom=1.0, iso_band=0.35,
                   prom_window=4, iso_window=10)

    highs = [s for s in swings if s.kind == "H"]
    lows = [s for s in swings if s.kind == "L"]
    assert [s.idx for s in highs] == [25]
    assert [s.idx for s in lows] == [45]
    assert highs[0].price == 106.0
    assert lows[0].price == 94.0
    assert highs[0].confirm_idx == 27
    assert lows[0].confirm_idx == 47


def test_pivot_ties_fail_the_strict_filter():
    df = _frame()
    df.loc[df.index[25], ["high", "low", "close"]] = [106.0, 100.0, 101.0]
    df.loc[df.index[26], ["high", "low", "close"]] = [106.0, 100.0, 101.0]

    swings = _scan(df, n=2, prom=1.0, iso_band=0.35,
                   prom_window=4, iso_window=10)

    assert not [s for s in swings if s.kind == "H" and s.price == 106.0]


def test_nearby_equal_extremes_are_rejected_by_isolation():
    df = _frame()
    df.loc[df.index[25], ["high", "low", "close"]] = [106.0, 100.0, 101.0]
    df.loc[df.index[28], ["high", "low", "close"]] = [105.9, 100.0, 101.0]

    swings = _scan(df, n=2, prom=1.0, iso_band=0.35,
                   prom_window=4, iso_window=10)

    assert not [s for s in swings if s.kind == "H" and 105.0 <= s.price <= 106.0]


def test_1h_and_5m_views_use_the_real_swing_detector():
    config = Config()
    assert set(config.real_swing_params) >= {60, 5, 1}
    assert config.real_swing_params[1]["n"] == 6

    df = _frame(120)
    df.loc[df.index[60], ["high", "low", "close"]] = [106.0, 100.0, 101.0]

    for tf in (60, 5, 1):
        view = build_view(df, tf, config)
        expected = real_swings(df, atr(df, config.atr_period),
                               **config.real_swing_params[tf])
        assert [(s.idx, s.kind, s.price) for s in view.major_swings] == \
               [(s.idx, s.kind, s.price) for s in expected]
        assert any(s.kind == "H" and s.price == 106.0 for s in view.major_swings)


def test_bos_long_at_a_downtrend_extreme(monkeypatch):
    df = _frame(30)
    df["close"] = 100.0
    df.loc[df.index[19], "close"] = 109.0
    df.loc[df.index[20], "close"] = 111.0   # above the LH at 110

    swings = [
        Swing(2, df.index[2], 80.0, "L", 4),
        Swing(5, df.index[5], 120.0, "H", 7),
        Swing(8, df.index[8], 70.0, "L", 10),     # final LL
        Swing(10, df.index[10], 110.0, "H", 12),  # LH
    ]
    monkeypatch.setattr("botv2.features.swing_points", lambda frame, k: swings)

    events = bos_events(df)

    assert [(e.direction, e.kind, e.level, e.label) for e in events] == [
        (1, "BOS", 110.0, "LH"),
    ]
    assert events[0].idx == 20


def test_bos_short_at_an_uptrend_extreme(monkeypatch):
    df = _frame(32)
    df["close"] = 100.0
    df.loc[df.index[29], "close"] = 101.0
    df.loc[df.index[30], "close"] = 99.0    # below the HL at 100

    swings = [
        Swing(2, df.index[2], 120.0, "H", 4),
        Swing(5, df.index[5], 80.0, "L", 7),
        Swing(8, df.index[8], 130.0, "H", 10),   # final HH
        Swing(10, df.index[10], 100.0, "L", 12), # HL
    ]
    monkeypatch.setattr("botv2.features.swing_points", lambda frame, k: swings)

    events = bos_events(df)

    assert [(e.direction, e.kind, e.level, e.label) for e in events] == [
        (-1, "BOS", 100.0, "HL"),
    ]
    assert events[0].idx == 30


def test_1m_choch_follows_reversal_sequences(monkeypatch):
    df = _frame(40)
    df["close"] = 100.0
    df.loc[df.index[19], "close"] = 99.0
    df.loc[df.index[20], "close"] = 101.0   # above the high at 100
    df.loc[df.index[30], "close"] = 111.0
    df.loc[df.index[31], "close"] = 109.0   # below the low at 110

    swings = [
        Swing(1, df.index[1], 95.0, "L", 3),
        Swing(3, df.index[3], 105.0, "H", 5),
        Swing(5, df.index[5], 90.0, "L", 7),       # final LL
        Swing(10, df.index[10], 100.0, "H", 12),
        Swing(15, df.index[15], 95.0, "L", 17),    # higher low
        Swing(22, df.index[22], 120.0, "H", 24),   # final HH
        Swing(25, df.index[25], 110.0, "L", 27),
        Swing(28, df.index[28], 115.0, "H", 30),   # lower high
    ]
    monkeypatch.setattr("botv2.features.swing_points", lambda frame, k: swings)

    events = choch_events(df)

    assert [(e.direction, e.kind, e.level, e.label) for e in events] == [
        (1, "CHoCH", 100.0, "HL"),
        (-1, "CHoCH", 110.0, "LH"),
    ]
    assert [e.idx for e in events] == [20, 31]


def test_bos_uses_only_unswept_lh(monkeypatch):
    df = _frame(30)
    df["close"] = 100.0
    df.loc[df.index[15], "high"] = 115.0   # earlier wick sweeps the LH at 110
    df.loc[df.index[15], "close"] = 105.0
    df.loc[df.index[19], "close"] = 105.0
    df.loc[df.index[20], "close"] = 112.0

    swings = [
        Swing(2, df.index[2], 80.0, "L", 4),
        Swing(5, df.index[5], 120.0, "H", 7),
        Swing(8, df.index[8], 70.0, "L", 10),
        Swing(10, df.index[10], 110.0, "H", 12),  # LH, but swept at bar 15
    ]
    monkeypatch.setattr("botv2.features.swing_points", lambda frame, k: swings)

    assert bos_events(df) == []
    assert bos_reference(df, swings, 1) == (None, "")


def test_rsi_confirmation_requires_a_preceding_divergence():
    from types import SimpleNamespace

    from botv2.analysis import _rsi_confirm
    from botv2.features import Divergence

    idx = pd.date_range("2026-01-01", periods=10, freq="5min", tz="UTC")
    rsi = pd.Series([50.0] * 10, index=idx)
    rsi.iloc[4] = 30.0
    rsi.iloc[7] = 40.0
    swings = [
        Swing(1, idx[1], 100.0, "L", 2),
        Swing(4, idx[4], 95.0, "L", 5),   # divergence ends here
        Swing(7, idx[7], 97.0, "L", 8),   # HL price + HL RSI = confirmation
    ]
    divergence = Divergence(5, idx[5], 1, 100.0, 95.0, 25.0, 30.0, False)

    with_div = SimpleNamespace(structure=SimpleNamespace(swings=swings),
                               rsi=rsi, divergences=[divergence])
    without = SimpleNamespace(structure=SimpleNamespace(swings=swings),
                              rsi=rsi, divergences=[])

    assert _rsi_confirm(with_div, 1) is True
    assert _rsi_confirm(without, 1) is False


def test_live_1m_candle_replaces_only_the_current_minute(monkeypatch):
    from botv2.charts import _merge_live_candle

    candles = [
        {"time": 100, "open": 1, "high": 2, "low": 1, "close": 2},
        {"time": 160, "open": 2, "high": 3, "low": 2, "close": 3},
        {"time": 220, "open": 3, "high": 4, "low": 3, "close": 3.5},
    ]
    live = {"time": 220, "open": 3, "high": 4.5, "low": 2.5, "close": 4.2}
    monkeypatch.setattr("botv2.charts.live_1m_candle",
                        lambda config, symbol="XAUUSD": live)

    kept = _merge_live_candle(candles, Config())
    assert [c["time"] for c in kept] == [100, 160, 220]
    assert kept[-1] is live

    live_new = {"time": 280, "open": 4, "high": 5, "low": 4, "close": 4.8}
    monkeypatch.setattr("botv2.charts.live_1m_candle",
                        lambda config, symbol="XAUUSD": live_new)
    added = _merge_live_candle(candles, Config())
    assert [c["time"] for c in added] == [100, 160, 220, 280]
    assert added[-1] is live_new


def test_range_breakout_from_a_tight_range():
    df = _frame(45)
    df["close"] = 100.0
    df["high"] = 100.5
    df["low"] = 99.5
    df.loc[df.index[34], "close"] = 100.2
    df.loc[df.index[35], ["open", "high", "low", "close"]] = [100.2, 102.0, 100.0, 101.8]

    events = range_breakout_events(df, atr(df, 14))
    ups = [e for e in events if e.direction > 0]
    assert ups
    assert ups[-1].kind == "Range breakout"
    assert ups[-1].idx == 35
    assert ups[-1].level == 100.5


def test_hl_sweep_reversal_long():
    df = _frame(30)
    df["close"] = 100.0
    df.loc[df.index[20], "low"] = 94.0      # sweep the HL at 95
    df.loc[df.index[21], "close"] = 106.0   # close above H2 at 105

    swings = [
        Swing(1, df.index[1], 90.0, "L", 3),
        Swing(5, df.index[5], 100.0, "H", 7),
        Swing(10, df.index[10], 95.0, "L", 12),   # HL
        Swing(15, df.index[15], 105.0, "H", 17),  # H2
    ]
    structure = Structure(1, swings, [], swings[-2], swings[-1], swings)

    events = sweep_reversal_events(df, structure)
    assert len(events) == 1
    event = events[0]
    assert (event.kind, event.direction, event.level, event.swept) == (
        "HL sweep", 1, 105.0, 95.0)
    assert event.idx == 21


def test_box_range_detector_and_breakout():
    df = _frame(90)
    df["high"] = 101.0
    df["low"] = 99.0
    df["open"] = 100.0
    df["close"] = 100.0 + (np.arange(90) % 2) * 0.1 - 0.05
    df.loc[df.index[80], ["open", "high", "low", "close"]] = [100.0, 104.0, 100.0, 103.5]

    ranges = detect_ranges(df, rolling_atr(df, 14))
    assert ranges
    box = ranges[-1]
    assert box.break_dir == "up"
    assert box.high == 101.0
    assert box.low == 99.0
    assert box.break_idx == 80


def test_range_resolver_sweep_and_breakout():
    atr_series = pd.Series([1.0] * 10, index=pd.date_range(
        "2026-01-01", periods=10, freq="5min", tz="UTC"))

    sweep = _frame(10)
    sweep["open"] = 100.0
    sweep["close"] = 100.0
    sweep["high"] = 100.4
    sweep["low"] = 99.6
    sweep.loc[sweep.index[0], ["high", "close"]] = [101.2, 100.4]
    sweep["atr"] = 1.0

    event = resolve_range_event(sweep, atr_series, 101.0, 99.0, 0)
    assert event["type"] == "sweep"
    assert event["side"] == "up"
    assert event["trade"] == "short"

    breakout = _frame(10)
    breakout["open"] = 100.0
    breakout["close"] = 102.5
    breakout["high"] = 102.6
    breakout["low"] = 99.6
    breakout["atr"] = 1.0
    breakout.loc[breakout.index[0], ["high", "close"]] = [101.2, 101.2]

    event = resolve_range_event(breakout, atr_series, 101.0, 99.0, 0)
    assert event["type"] == "breakout"
    assert event["side"] == "up"
    assert event["trade"] == "long"
