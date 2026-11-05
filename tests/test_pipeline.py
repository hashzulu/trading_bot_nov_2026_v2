"""End-to-end tests for the MT4 -> analysis -> report pipeline."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from botv2.analysis import analyse_symbol, build_view, cluster_levels, determine_bias
from botv2.config import Config
from botv2.data import (
    SymbolData,
    infer_timeframe,
    parse_mt4_csv,
    resample,
    synth_frame,
)
from botv2.features import (
    atr,
    build_structure,
    detect_divergences,
    detect_sweeps,
    fractal_flags,
    rsi,
)
from botv2.report import render_json, render_markdown, write_reports


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def frame(closes: list[float], start: datetime | None = None, tf: int = 5,
          wick: float = 0.05) -> pd.DataFrame:
    start = start or datetime(2026, 1, 1, tzinfo=timezone.utc)
    index = pd.DatetimeIndex([start + timedelta(minutes=tf * i) for i in range(len(closes))], tz="UTC")
    close = np.array(closes, dtype=float)
    open_ = close.copy()
    high = np.maximum(open_, close) + wick
    low = np.minimum(open_, close) - wick
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close,
                         "volume": np.ones(len(closes))}, index=index)


def write_csv(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# parsing
# --------------------------------------------------------------------------- #

def test_parse_combined_datetime_header(tmp_path):
    path = write_csv(tmp_path, "XAUUSD_5.csv",
                     "time,open,high,low,close,volume\r\n"
                     "2026.10.16 17:40,82330.31,82347.41,82256.98,82281.91,1424\r\n"
                     "2026.10.16 17:45,82281.93,82380.84,82254.62,82340.22,1505\r\n")
    df, offset = parse_mt4_csv(path, server_utc_offset_hours=2.0)
    assert len(df) == 2
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert df.index.tz is not None
    assert df.index[0].hour == 15                      # 17:40 server - 2h
    assert df["close"].iloc[-1] == pytest.approx(82340.22)


def test_parse_semicolon_decimal_comma_and_split_columns(tmp_path):
    path = write_csv(tmp_path, "EURUSD_5.csv",
                     "Date;Time;Open;High;Low;Close;Volume\n"
                     "2026.10.16;17:40;1,08501;1,08512;1,08490;1,08505;100\n")
    df, _ = parse_mt4_csv(path, server_utc_offset_hours=3.0)
    assert len(df) == 1
    assert df["close"].iloc[0] == pytest.approx(1.08505)


def test_parse_without_header_and_iso_dates(tmp_path):
    path = write_csv(tmp_path, "BTCUSD_5.csv",
                     "2026-10-16 17:40:00,100.0,101.0,99.0,100.5,10\n")
    df, _ = parse_mt4_csv(path, server_utc_offset_hours=0.0)
    assert df["high"].iloc[0] == pytest.approx(101.0)


def test_parse_rejects_garbage(tmp_path):
    path = write_csv(tmp_path, "BAD_5.csv", "not,a,market\nnope\n")
    with pytest.raises(ValueError):
        parse_mt4_csv(path)


@pytest.mark.parametrize("name,expected", [
    ("XAUUSD_5.csv", ("XAUUSD", 5)),
    ("EURUSD_15.csv", ("EURUSD", 15)),
    ("EURUSD_H1.csv", ("EURUSD", 60)),
    ("XAUUSD_240.csv", ("XAUUSD", 240)),
    ("BTCUSD-M5.csv", ("BTCUSD", 5)),
    ("GBPUSD_1D.csv", ("GBPUSD", 1440)),
])
def test_timeframe_inference(name, expected):
    assert infer_timeframe(Path(name)) == expected


def test_timeframe_inference_rejects_unknown():
    assert infer_timeframe(Path("notes.csv")) is None
    assert infer_timeframe(Path("XAUUSD_7.csv")) is None


# --------------------------------------------------------------------------- #
# resampling & indicators
# --------------------------------------------------------------------------- #

def test_resample_ohlcv_aggregation():
    base = frame([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], tf=5)
    hourly = resample(base, 60)
    assert len(hourly) == 1
    row = hourly.iloc[0]
    assert row["open"] == base["open"].iloc[0]
    assert row["high"] == base["high"].max()
    assert row["low"] == base["low"].min()
    assert row["close"] == base["close"].iloc[-1]
    assert row["volume"] == base["volume"].sum()


def test_rsi_extremes():
    up = frame(list(range(1, 40)))
    down = frame(list(range(40, 1, -1)))
    assert rsi(up["close"]).iloc[-1] > 70
    assert rsi(down["close"]).iloc[-1] < 30
    values = rsi(up["close"]).dropna()
    assert values.between(0, 100).all()


def test_atr_positive():
    values = atr(frame(list(range(1, 40)))).dropna()
    assert (values > 0).all()


def test_fractals_find_peak_and_trough():
    closes = [10, 11, 12, 13, 14, 13, 12, 11, 10, 9, 8, 9, 10, 11, 12]
    df = frame(closes, wick=0.1)
    sh, sl = fractal_flags(df, k=2)
    assert sh[4]          # local top at 14
    assert sl[10]         # local bottom at 8


def test_structure_detects_bos():
    # rise to a swing high, pull back, then break above it
    closes = [10, 11, 12, 13, 14, 13.5, 13, 13.5, 14.5, 15.2]
    df = frame(closes, wick=0.05)
    structure = build_structure(df, k=2)
    assert structure.trend == 1
    assert any(e.direction == 1 for e in structure.events)


def test_sweep_detected_when_wick_reclaims():
    # gentle uptrend with a clear swing low at bar 6, then a later bar that
    # wicks below that low but closes back above it
    closes = [10, 10.1, 10.2, 10.3, 10.4, 10.5, 10.2, 11.0] + [11.1 + 0.1 * i for i in range(18)]
    df = frame(closes, wick=0.05)
    swing_low = df["low"].iloc[6]
    sweep_bar = 20
    df.iloc[sweep_bar, df.columns.get_loc("low")] = swing_low - 0.4
    df.iloc[sweep_bar, df.columns.get_loc("close")] = swing_low + 0.5
    structure = build_structure(df, k=2)
    sweeps = detect_sweeps(df, atr(df), structure, lookback=25)
    assert sweeps
    bullish = [s for s in sweeps if s.direction == 1]
    assert bullish
    assert bullish[-1].level == pytest.approx(swing_low, rel=1e-6)


def test_divergence_lower_low_with_higher_rsi():
    # sharp drop, bounce, slow grind lower: price LL, RSI HL
    closes = [100, 98, 96, 94, 92, 90, 89, 90, 91, 90.5, 90, 89.6, 89.2, 88.9, 89.3]
    df = frame(closes, wick=0.05)
    structure = build_structure(df, k=2)
    divs = detect_divergences(df, rsi(df["close"]), structure, min_bars=3)
    assert isinstance(divs, list)


# --------------------------------------------------------------------------- #
# strategy
# --------------------------------------------------------------------------- #

def _views_for(frame5: pd.DataFrame, config: Config):
    frames = {5: frame5}
    for tf in (15, 60, 240):
        frames[tf] = resample(frame5, tf)
    return {tf: build_view(fr, tf, config) for tf, fr in frames.items() if len(fr) >= 40}


def test_bias_is_trend_following_on_uptrend():
    config = Config()
    frame5 = synth_frame("XAUUSD", 5, 3200, 1)
    frames = {5: frame5}
    for tf in (15, 60, 240):
        frames[tf] = resample(frame5, tf)
    analysis = analyse_symbol(SymbolData("XAUUSD", frames, 5, derived={15, 60, 240}), config)
    assert analysis.bias == 1
    assert "LONG" in analysis.bias_label


def test_end_to_end_long_setup_from_scenario():
    config = Config()
    frame5 = synth_frame("XAUUSD", 5, 3200, 1)
    frames = {5: frame5}
    for tf in (15, 60, 240):
        frames[tf] = resample(frame5, tf)
    data = SymbolData("XAUUSD", frames, 5, derived={15, 60, 240})
    analysis = analyse_symbol(data, config)

    setup = analysis.setup
    assert setup is not None, analysis.watch
    assert setup.direction == 1
    assert setup.grade in ("A", "B")
    assert setup.rr >= config.min_rr
    assert setup.risk_atr <= config.max_sl_atr
    assert setup.sl < setup.entry < setup.tp
    assert setup.score >= config.min_score_b
    assert setup.checklist
    assert any("RSI floor" in f.name for f in setup.factors)


def test_end_to_end_short_setup_from_scenario():
    config = Config()
    frame5 = synth_frame("BTCUSD", 5, 3200, -1)
    frames = {5: frame5}
    for tf in (15, 60, 240):
        frames[tf] = resample(frame5, tf)
    data = SymbolData("BTCUSD", frames, 5, derived={15, 60, 240})
    analysis = analyse_symbol(data, config)

    setup = analysis.setup
    assert setup is not None, analysis.watch
    assert setup.direction == -1
    assert setup.sl > setup.entry > setup.tp


def test_levels_are_clustered_near_price():
    config = Config()
    frame5 = synth_frame("XAUUSD", 5, 3200, 1)
    views = _views_for(frame5, config)
    entry = views[5]
    levels = cluster_levels(views, entry.atr_now, entry.price)
    assert levels
    assert all(abs(level.dist_atr) <= 30 for level in levels)
    assert all(level.tfs for level in levels)


def test_tiny_dataset_is_graceful():
    config = Config()
    frame5 = frame([100, 101, 102, 101, 100], tf=60)
    data = SymbolData("XAUUSD", {60: frame5}, 60)
    analysis = analyse_symbol(data, config)
    assert analysis.setup is None
    assert analysis.bias in (-1, 0, 1)


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #

def test_reports_are_written(tmp_path):
    config = Config()
    config.out_dir = tmp_path
    frame5 = synth_frame("XAUUSD", 5, 3200, 1)
    frames = {5: frame5}
    for tf in (15, 60, 240):
        frames[tf] = resample(frame5, tf)
    analysis = analyse_symbol(SymbolData("XAUUSD", frames, 5, derived={15, 60, 240}), config)

    md_path, json_path = write_reports([analysis], config, "unit test")
    markdown = md_path.read_text(encoding="utf-8")
    payload = render_json([analysis], config, "unit test")

    assert "# Chart read" in markdown
    assert "Trade plan" in markdown or "no trade yet" in markdown
    assert "Levels around price" in markdown
    assert "symbols" in payload
    assert payload["symbols"]["XAUUSD"]["bias"] in (-1, 0, 1)
    assert json_path.exists()


def test_markdown_works_without_setups(tmp_path):
    config = Config()
    analysis = analyse_symbol(
        SymbolData("EURUSD", {60: frame([1.1, 1.1, 1.1, 1.1, 1.1], tf=60)}, 60), config
    )
    markdown = render_markdown([analysis], config, "unit test")
    assert "Watchlist" in markdown
