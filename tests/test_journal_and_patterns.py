"""Tests for the journal calibration layer and 1-min pattern detection."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from botv2.analysis import (
    WatchItem,
    analyse_symbol,
    build_view,
    detect_setup,
    determine_bias,
    potential_entries,
)
from botv2.config import Config
from botv2.data import SymbolData, resample, synth_frame
from botv2.features import atr, build_structure, detect_head_shoulders
from botv2.journal import (
    TOP_RULES,
    JournalProfile,
    apply_calibration,
    build_profile,
    render_lessons_markdown,
    setup_history,
)
from botv2.narrative import headline
from botv2.report import (
    _evaluate_plan_status,
    _save_active_plans,
    apply_active_plans,
    enforce_one_new_recommendation,
    load_active_plans,
    recommendation_signature,
    record_active_plan,
    render_markdown,
    set_plan_result,
    suppress_duplicate_setups,
)
from botv2.serve import mark_plan
from botv2.dashboard import _cet_time


def frame(closes: list[float], wick: float = 0.1, tf: int = 1) -> pd.DataFrame:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    index = pd.DatetimeIndex([start + timedelta(minutes=tf * i) for i in range(len(closes))], tz="UTC")
    close = np.array(closes, dtype=float)
    return pd.DataFrame({
        "open": close, "high": close + wick, "low": close - wick,
        "close": close, "volume": np.ones(len(closes)),
    }, index=index)


# --------------------------------------------------------------------------- #
# head & shoulders
# --------------------------------------------------------------------------- #

def topping_pattern() -> pd.DataFrame:
    closes = ([100, 101, 102, 103, 104, 105] +          # left shoulder
              [104, 103, 102, 101, 100] +               # trough 1
              [101, 103, 105, 107, 109] +               # head
              [108, 106, 104, 102, 101] +               # trough 2
              [102, 104, 105, 105.5] +                  # right shoulder
              [104, 102, 100, 98, 97])                  # neckline break
    return frame(closes)


def bottoming_pattern() -> pd.DataFrame:
    closes = ([110, 109, 108, 107, 106, 105] +
              [106, 107, 108, 109, 110] +
              [109, 107, 105, 103, 101] +
              [102, 104, 106, 108, 109] +
              [108, 106, 105, 104.5] +
              [106, 108, 110, 112, 113, 114])
    return frame(closes)


def test_topping_hs_detected_and_complete():
    df = topping_pattern()
    structure = build_structure(df, k=1)
    patterns = detect_head_shoulders(df, structure, atr(df), lookback=80, break_window=20)
    assert patterns
    pattern = patterns[-1]
    assert pattern.direction == -1
    assert pattern.complete
    assert pattern.neckline < pattern.head_price
    assert pattern.left_idx < pattern.head_idx < pattern.right_idx


def test_bottoming_inverse_hs_detected():
    df = bottoming_pattern()
    structure = build_structure(df, k=1)
    patterns = detect_head_shoulders(df, structure, atr(df), lookback=80, break_window=20)
    assert any(p.direction == 1 for p in patterns)


def test_hs_without_break_is_incomplete():
    df = topping_pattern().iloc[:-6]      # cut before the neckline break
    structure = build_structure(df, k=1)
    patterns = detect_head_shoulders(df, structure, atr(df), lookback=80, break_window=20)
    assert all(not p.complete for p in patterns) or not patterns


# --------------------------------------------------------------------------- #
# journal profile
# --------------------------------------------------------------------------- #

def make_journal_db(path: Path) -> Path:
    con = sqlite3.connect(path)
    con.execute("""create table trades (
        id integer primary key, date text, number integer, symbol text, direction text,
        setup text, watching text, thesis text, notes text, execution text, tags text,
        pnl real, r real, accountId integer, createdAt text)""")
    con.execute("""create table journalPerTradeCompliance (
        tradeId integer, accountId integer, updatedAt text, watchingRsi integer,
        rsiDivergence integer, entryPattern1m__hs1m integer, entryPattern1m__choc1m integer,
        sweptSwing integer)""")
    con.execute("create table accounts (id integer, name text, rules text, preTradeRules text)")
    con.execute("""insert into accounts values (1, 'MASTER',
        '[{"label":"Max 3/day"},{"label":"Never ++ SL"}]',
        '[{"label":"5-min RSI Divergence"},{"label":"1-min entry pattern"}]')""")

    trades = [
        (1, "2026-09-01", 1, "XAUUSD", "Long",
         "Entered on a bullish 1-min head&shoulders", "5-min RSI floor at 26",
         "thesis", "NA", "Entered on a 1-min H&S; SL 1R; hit TP", '["continuation_long"]', 100.0, 1.0, 1,
         "2026-09-01T13:00:00Z"),
        (2, "2026-09-01", 2, "XAUUSD", "Long",
         "Entered on a bullish 1-min BOS", "5-min RSI floor at 24",
         "thesis", "NA", "Increased SL to 1.8R as trade moved against me; took out", '["continuation_long"]', -200.0, -1.8, 1,
         "2026-09-01T14:00:00Z"),
        (3, "2026-09-02", 1, "EURUSD", "Short",
         "Entered on a bearish 1-min CHoC", "5-min RSI ceiling at 74",
         "thesis", "NA", "Entered an incomplete pattern; stopped", "[]", -150.0, -1.0, 1,
         "2026-09-02T10:00:00Z"),
        (4, "2026-09-03", 1, "XAUUSD", "Long",
         "Entered on a bullish 1-min range sweep", "5-min RSI floor at 20",
         "thesis", "NA", "Entered on a range sweep; closed at 2R", "[]", 300.0, 2.0, 1,
         "2026-09-03T15:00:00Z"),
        (5, "2026-09-04", 1, "XAUUSD", "Short",
         "Entered on a bearish 1-min BOS", "5-min RSI ceiling at 78",
         "thesis", "NA", "Moved SL to -2R as price moved against me; took out", "[]", -250.0, -2.0, 1,
         "2026-09-04T16:00:00Z"),
    ]
    con.executemany("insert into trades values (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", trades)
    compliance = [
        (1, 1, "2026-09-01", 1, 1, 1, 0, 0),
        (2, 1, "2026-09-01", 1, 1, 0, 0, 0),
        (3, 1, "2026-09-02", 0, 0, 0, 1, 0),
        (4, 1, "2026-09-03", 1, 1, 0, 0, 1),
        (5, 1, "2026-09-04", 0, 0, 0, 0, 0),
    ]
    con.executemany("insert into journalPerTradeCompliance values (?,?,?,?,?,?,?,?)", compliance)
    con.commit()
    con.close()
    return path


def test_profile_extracts_edges_leaks_and_zones(tmp_path):
    db = make_journal_db(tmp_path / "TradingJournalDB.sqlite")
    profile = build_profile(db)
    assert profile.trades == 5
    assert profile.win_rate == pytest.approx(0.4)
    assert any("Max 3/day" == rule for rule in profile.rules)
    assert "5-min RSI Divergence" in profile.pretrade_rules

    leak = profile.mistake("widened_stop")
    assert leak is not None and leak.pnl < 0

    floor_zone = profile.zone("XAUUSD", "long")
    assert floor_zone is not None and 20 <= floor_zone.median <= 26

    markdown = render_lessons_markdown(profile)
    assert "Journal profile" in markdown and "Leaks" in markdown


def test_calibration_bends_thresholds(tmp_path):
    db = make_journal_db(tmp_path / "TradingJournalDB.sqlite")
    profile = build_profile(db)
    config = Config()
    config.symbols = ("XAUUSD", "EURUSD")
    apply_calibration(config, profile)
    assert config.rsi_floor_for("XAUUSD") <= config.rsi_floor
    assert config.rsi_ceiling_for("EURUSD") >= config.rsi_ceiling


def test_setup_history_maps_factors():
    profile = JournalProfile(
        trades=10, win_rate=0.5,
        factors=[],
    )
    from botv2.journal import FactorStat
    profile.factors = [FactorStat("rsiDivergence", "5-min RSI divergence", 20, 0.6, 500, 0.4, 100)]
    stats = setup_history(profile, ["RSI divergence (price LL / RSI HL)"])
    assert stats and stats[0].key == "rsiDivergence"


def test_report_includes_journal_section(tmp_path):
    db = make_journal_db(tmp_path / "TradingJournalDB.sqlite")
    profile = build_profile(db)
    config = Config()
    config.out_dir = tmp_path
    frame5 = synth_frame("XAUUSD", 5, 3200, 1)
    frames = {5: frame5}
    for tf in (15, 60, 240):
        frames[tf] = resample(frame5, tf)
    analysis = analyse_symbol(SymbolData("XAUUSD", frames, 5, derived={15, 60, 240}), config)
    markdown = render_markdown([analysis], config, "unit test", profile=profile)
    assert "Journal calibration" in markdown
    assert "RSI zones you actually watched" in markdown


def test_hs_trigger_is_used_as_entry_pattern():
    """A completed inverse H&S neckline break should be the trigger and score points."""
    closes = ([112, 111] +
              [110, 109, 108, 107, 106, 105] +
              [106, 107, 108, 109, 110] +
              [109, 107, 105, 103, 101] +
              [102, 104, 106, 108, 109] +
              [108, 106, 105, 104.5] +
              [106, 108, 110])
    entry_frame = frame(closes)
    # context frame: an uptrend so the bias is not contradicted
    context_frame = frame([100 + 0.05 * i for i in range(400)], tf=60)

    config = Config()
    config.auto_entry_tf = False
    # This test isolates the H&S trigger; the journal rule gate is covered
    # separately (and RSI confirmation now requires a preceding divergence).
    config.min_top_rules = 1
    views = {
        60: build_view(context_frame, 60, config),
        1: build_view(entry_frame, 1, config),
    }
    setup, watch = detect_setup("XAUUSD", views, 1, "LONG (trend)", [], config, entry_tf=1)
    assert setup is not None, watch
    assert "H&S" in setup.trigger_kind
    assert any("H&S" in factor.name for factor in setup.factors)
    assert setup.sl < setup.entry < setup.tp


def test_rule_gate_blocks_when_rules_cannot_be_met():
    closes = ([112, 111] +
              [110, 109, 108, 107, 106, 105] +
              [106, 107, 108, 109, 110] +
              [109, 107, 105, 103, 101] +
              [102, 104, 106, 108, 109] +
              [108, 106, 105, 104.5] +
              [106, 108, 110])
    entry_frame = frame(closes)
    context_frame = frame([100 + 0.05 * i for i in range(400)], tf=60)
    config = Config()
    config.auto_entry_tf = False
    config.min_top_rules = 6          # impossible: at most 5 verifiable here
    views = {60: build_view(context_frame, 60, config),
             1: build_view(entry_frame, 1, config)}
    setup, watch = detect_setup("XAUUSD", views, 1, "LONG (trend)", [], config, entry_tf=1)
    assert setup is None
    assert watch is not None
    assert any("pre-trade rules" in item for item in watch.missing)


def test_calibrated_setup_reports_rule_fields():
    config = Config()
    frame5 = synth_frame("XAUUSD", 5, 3200, 1)
    frames = {5: frame5}
    for tf in (15, 60, 240):
        frames[tf] = resample(frame5, tf)
    analysis = analyse_symbol(SymbolData("XAUUSD", frames, 5, derived={15, 60, 240}), config)
    setup = analysis.setup
    assert setup is not None, analysis.watch
    assert setup.rule_count >= config.min_top_rules
    assert setup.rules_ticked
    assert set(setup.rules_ticked).issubset(set(TOP_RULES))
    markdown = render_markdown([analysis], config, "unit test")
    assert "Top rules" in markdown


def test_candle_patterns_and_reversal_candles():
    from botv2.features import detect_candle_patterns, detect_reversal_continuation

    closes = [100, 101, 102, 103, 104, 105, 104, 103, 102, 101, 100.2, 101.5, 102.5]
    df = frame(closes, wick=0.05)
    # make bar 10 a hammer: long lower wick, bullish close
    df.iloc[10, df.columns.get_loc("open")] = 101.5
    df.iloc[10, df.columns.get_loc("close")] = 101.9
    df.iloc[10, df.columns.get_loc("low")] = 100.4
    df.iloc[10, df.columns.get_loc("high")] = 102.0
    a = atr(df)
    patterns = detect_candle_patterns(df, a)
    assert any(p.kind == "hammer" for p in patterns) or patterns == []

    # uptrend pullback then a strong bullish reversal candle
    up = [100 + i * 0.2 for i in range(10)]
    pull = [102.0, 101.8, 101.6, 101.4]
    rev = [101.3]
    closes2 = up + pull + rev
    df2 = frame(closes2, wick=0.05)
    i = len(closes2) - 1
    df2.iloc[i, df2.columns.get_loc("open")] = 101.4
    df2.iloc[i, df2.columns.get_loc("close")] = 101.9
    df2.iloc[i, df2.columns.get_loc("low")] = 101.0
    df2.iloc[i, df2.columns.get_loc("high")] = 102.0
    hits = detect_reversal_continuation(df2, atr(df2), trend=1)
    assert isinstance(hits, list)


def test_setup_heading_shows_recommendation_time_in_cet(tmp_path):
    analysis = _synthetic_analysis(tmp_path)
    assert analysis.setup is not None
    config = Config()
    config.out_dir = tmp_path
    analysis.setup.recommended_at = datetime(2026, 11, 3, 14, 6, tzinfo=timezone.utc)
    text = render_markdown([analysis], config, generated=datetime.now(timezone.utc))
    assert "recommended 2026-11-03 15:06 CET" in text
    assert "| Recommended | 2026-11-03 15:06 CET |" in text


def test_recommendation_log_appends_and_dedupes(tmp_path):
    from botv2.report import REC_LOG_NAME, append_recommendation

    config = Config()
    config.out_dir = tmp_path
    frame5 = synth_frame("XAUUSD", 5, 3200, 1)
    frames = {5: frame5}
    for tf in (15, 60, 240):
        frames[tf] = resample(frame5, tf)
    analysis = analyse_symbol(SymbolData("XAUUSD", frames, 5, derived={15, 60, 240}), config)
    assert analysis.setup is not None

    assert append_recommendation(analysis, analysis.setup, config) is True
    path = tmp_path / REC_LOG_NAME
    text = path.read_text(encoding="utf-8")
    assert "# Recommended trades" in text
    assert analysis.setup.symbol in text and "THESIS" in text

    # same setup again -> no duplicate
    assert append_recommendation(analysis, analysis.setup, config) is False
    assert path.read_text(encoding="utf-8").count("<!-- sig:") == 1


def _synthetic_analysis(tmp_path: Path):
    config = Config()
    config.out_dir = tmp_path
    frame5 = synth_frame("XAUUSD", 5, 3200, 1)
    frames = {5: frame5}
    for tf in (15, 60, 240):
        frames[tf] = resample(frame5, tf)
    return analyse_symbol(SymbolData("XAUUSD", frames, 5, derived={15, 60, 240}), config)


def test_potential_entries_are_shared_by_tldr_and_chart(tmp_path):
    analysis = _synthetic_analysis(tmp_path)
    view = analysis.views[analysis.entry_tf]
    pe = potential_entries(view)
    # the journal snapshot (used by the TL;DR) and the 1-min chart both read these
    assert analysis.journal["entry_long_above"] == pe.long_above
    assert analysis.journal["entry_short_below"] == pe.short_below
    assert analysis.journal["entry_retest"] == pe.retest

    analysis.setup = None
    analysis.watch = WatchItem(analysis.bias or -1, ["missing"], "watching")
    text = headline(analysis)
    for level in (pe.long_above, pe.short_below, pe.retest):
        if level is not None:
            rendered = f"{level:.2f}" if level >= 10 else f"{level:.5f}"
            assert rendered in text


def test_active_plan_freezes_levels_and_survives_missing_setup(tmp_path):
    analysis = _synthetic_analysis(tmp_path)
    setup = analysis.setup
    assert setup is not None
    config = Config()
    config.out_dir = tmp_path
    original = (setup.entry, setup.sl, setup.tp, setup.rr)

    assert record_active_plan(analysis, setup, config) is True
    # next cycle: live detection recomputes different levels
    setup.entry -= 1.4
    setup.sl += 0.1
    setup.tp -= 0.2
    setup.rr = 1.0
    setup.locked_sig = ""
    assert apply_active_plans([analysis], config) == 1
    assert (setup.entry, setup.sl, setup.tp, setup.rr) == original
    assert setup.locked_sig

    # if live detection loses the setup, the pinned recommendation still shows
    analysis.setup = None
    analysis.watch = WatchItem(-1, ["missing"], "watching")
    assert apply_active_plans([analysis], config) == 1
    assert analysis.setup is not None
    assert (analysis.setup.entry, analysis.setup.sl, analysis.setup.tp) == original[:3]


def test_active_plan_status_hit_sl_and_hit_tp(tmp_path):
    analysis = _synthetic_analysis(tmp_path)
    setup = analysis.setup
    assert setup is not None
    config = Config()
    config.out_dir = tmp_path
    view = analysis.views[analysis.entry_tf]
    recent = view.frame.tail(3)
    created = (view.frame.index[-3] - timedelta(minutes=analysis.entry_tf)).isoformat()
    base = {
        "symbol": "XAUUSD", "direction": setup.direction, "created": created,
        "status": "recommended", "trigger_tf": analysis.entry_tf,
    }
    if setup.direction > 0:
        hit_sl = {**base, "sl": float(recent["low"].min()) + 0.01, "tp": float(recent["high"].max()) + 10}
        hit_tp = {**base, "sl": float(recent["low"].min()) - 10, "tp": float(recent["high"].min()) - 0.01}
    else:
        hit_sl = {**base, "sl": float(recent["high"].max()) - 0.01, "tp": float(recent["low"].min()) - 10}
        hit_tp = {**base, "sl": float(recent["high"].max()) + 10, "tp": float(recent["low"].max()) + 0.01}
    assert _evaluate_plan_status(hit_sl, analysis, config) == "hit_sl"
    assert _evaluate_plan_status(hit_tp, analysis, config) == "hit_tp"


def test_active_plan_stores_snapshot_and_keeps_history(tmp_path):
    analysis = _synthetic_analysis(tmp_path)
    setup = analysis.setup
    assert setup is not None
    config = Config()
    config.out_dir = tmp_path
    assert record_active_plan(analysis, setup, config) is True

    plans = load_active_plans(config)
    assert len(plans) == 1
    key, plan = next(iter(plans.items()))
    assert key == plan["sig"]
    assert set(plan["snapshot"]) == {"tldr", "market_context", "watchlist"}
    assert plan["snapshot"]["tldr"]

    assert mark_plan(tmp_path, key, "hit_tp") is True
    closed = load_active_plans(config)[key]
    assert closed["status"] == "hit_tp"
    assert closed["closed_at"]
    assert closed["pnl_points"] == abs(plan["tp"] - plan["entry"])
    assert closed["pnl_r"] == abs(plan["tp"] - plan["entry"]) / abs(plan["entry"] - plan["sl"])
    assert closed["risk_usd"] == 300.0
    assert closed["pnl_usd"] == closed["pnl_r"] * 300.0

    # a genuinely new setup (different trigger level) is added, old history remains
    setup.trigger_level += 1.0
    setup.locked_sig = ""
    assert record_active_plan(analysis, setup, config) is True
    assert len(load_active_plans(config)) == 2


def test_recommendation_signature_ignores_drifted_levels(tmp_path):
    """Re-detecting the same trigger must not look like a new trade."""
    analysis = _synthetic_analysis(tmp_path)
    setup = analysis.setup
    assert setup is not None
    before = recommendation_signature(analysis, setup)
    setup.entry += 2.0
    setup.sl += 0.5
    setup.tp -= 0.5
    setup.grade = "A" if setup.grade != "A" else "B"
    assert recommendation_signature(analysis, setup) == before


def test_auto_detected_result_is_persisted(tmp_path):
    """A TP/SL the engine detects itself must survive to active_plans.json."""
    analysis = _synthetic_analysis(tmp_path)
    setup = analysis.setup
    assert setup is not None
    config = Config()
    config.out_dir = tmp_path
    assert record_active_plan(analysis, setup, config) is True

    plans = load_active_plans(config)
    key = next(iter(plans))
    plan = plans[key]
    view = analysis.views[analysis.entry_tf]
    recent = view.frame.tail(3)
    plan["created"] = (view.frame.index[-3] - timedelta(minutes=analysis.entry_tf)).isoformat()
    plan["status"] = "open"
    plan["opened_at"] = plan["created"]
    if setup.direction > 0:
        plan["sl"] = float(recent["low"].min()) + 0.01
        plan["tp"] = float(recent["high"].max()) + 10
    else:
        plan["sl"] = float(recent["high"].max()) - 0.01
        plan["tp"] = float(recent["low"].min()) - 10
    _save_active_plans(plans, config)

    analysis.setup = None
    apply_active_plans([analysis], config)

    saved = load_active_plans(config)[key]
    assert saved["status"] == "hit_sl"
    assert saved["closed_at"]
    assert saved["pnl_r"] == pytest.approx(-1.0)
    assert saved["pnl_usd"] == pytest.approx(-300.0)


def test_fill_only_counts_candles_after_the_signal_candle():
    from types import SimpleNamespace
    import pandas as pd

    index = pd.DatetimeIndex([
        datetime(2026, 11, 3, 10, 0, tzinfo=timezone.utc),
        datetime(2026, 11, 3, 10, 1, tzinfo=timezone.utc),
        datetime(2026, 11, 3, 10, 2, tzinfo=timezone.utc),
    ])
    frame = pd.DataFrame({
        "open": [99.5, 99.3, 99.2],
        "high": [100.0, 99.8, 99.7],
        "low": [99.0, 99.1, 99.0],
        "close": [99.5, 99.3, 99.2],
        "volume": [1.0, 1.0, 1.0],
    }, index=index)
    analysis = SimpleNamespace(views={1: SimpleNamespace(frame=frame)}, entry_tf=1)
    config = Config()
    config.active_plan_max_age_hours = 100000.0
    plan = {
        "status": "recommended", "direction": -1,
        "entry": 99.9, "sl": 101.0, "tp": 98.0,
        "trigger_tf": 1,
        "created": datetime(2026, 11, 3, 10, 0, 30, tzinfo=timezone.utc).isoformat(),
        "signal_bar": index[0].isoformat(),
    }

    # the signal candle touched 99.9, but fills only count on the candles after it
    assert _evaluate_plan_status(plan, analysis, config) == "recommended"

    # the next candle trades the entry -> the plan opens
    frame.iloc[1, frame.columns.get_loc("high")] = 100.2
    assert _evaluate_plan_status(plan, analysis, config) == "open"
    assert plan["opened_at"] == index[1].isoformat()


def test_plan_opens_when_entry_is_touched(tmp_path):
    """A recommendation becomes OPEN once price trades its entry level."""
    analysis = _synthetic_analysis(tmp_path)
    setup = analysis.setup
    assert setup is not None
    config = Config()
    config.out_dir = tmp_path
    assert record_active_plan(analysis, setup, config) is True

    plans = load_active_plans(config)
    key = next(iter(plans))
    plan = plans[key]
    view = analysis.views[analysis.entry_tf]
    recent = view.frame.tail(3)
    plan["created"] = (view.frame.index[-3] - timedelta(minutes=analysis.entry_tf)).isoformat()
    plan["signal_bar"] = plan["created"]
    if setup.direction < 0:
        plan["entry"] = float(recent["high"].max()) - 0.01
        plan["sl"] = float(recent["high"].max()) + 10
        plan["tp"] = float(recent["low"].min()) - 10
    else:
        plan["entry"] = float(recent["low"].min()) + 0.01
        plan["sl"] = float(recent["low"].min()) - 10
        plan["tp"] = float(recent["high"].max()) + 10
    _save_active_plans(plans, config)

    analysis.setup = None
    apply_active_plans([analysis], config)

    saved = load_active_plans(config)[key]
    assert saved["status"] == "open"
    assert saved["opened_at"]


def test_recommendation_cooldown_blocks_back_to_back_trades(tmp_path):
    analysis = _synthetic_analysis(tmp_path)
    setup = analysis.setup
    assert setup is not None
    config = Config()
    config.out_dir = tmp_path
    config.recommendation_cooldown_minutes = 10
    assert record_active_plan(analysis, setup, config) is True

    # a different/refreshed setup immediately after the recommendation is dropped
    setup.locked_sig = ""
    assert enforce_one_new_recommendation([analysis], config) == 1
    assert analysis.setup is None
    assert any("no new trade" in note for note in analysis.notes)

    # after the window the same setup may be recommended again
    plans = load_active_plans(config)
    key = next(iter(plans))
    plans[key]["created"] = (datetime.now(timezone.utc) - timedelta(minutes=11)).isoformat()
    _save_active_plans(plans, config)
    analysis.setup = setup
    analysis.notes = []
    assert enforce_one_new_recommendation([analysis], config) == 0
    assert analysis.setup is setup


def test_open_trade_blocks_a_different_setup_until_it_closes(tmp_path):
    """One open trade at a time: a new trigger is ignored while a plan is live."""
    analysis = _synthetic_analysis(tmp_path)
    setup = analysis.setup
    assert setup is not None
    config = Config()
    config.out_dir = tmp_path
    assert record_active_plan(analysis, setup, config) is True
    # keep the plan open for the duration of the test
    plans = load_active_plans(config)
    open_key = next(iter(plans))
    plans[open_key]["sl"] = float(setup.entry) + 50.0
    plans[open_key]["tp"] = float(setup.entry) - 50.0
    _save_active_plans(plans, config)

    # a genuinely different trigger appears while the first trade is open
    setup.trigger_level += 7.5
    setup.entry += 1.0
    setup.locked_sig = ""
    different_trigger = setup.trigger_level

    # the live plan owns the headline, so the new setup is replaced by it
    assert apply_active_plans([analysis], config) == 1
    assert analysis.setup.locked_sig
    assert analysis.setup.trigger_level != different_trigger

    # and a second plan cannot be recorded while the first one is live
    setup.trigger_level = different_trigger
    setup.locked_sig = ""
    assert record_active_plan(analysis, setup, config) is False
    assert len(load_active_plans(config)) == 1

    # once the open trade hits SL, a new setup can be recorded
    plans = load_active_plans(config)
    first_key = next(iter(plans))
    set_plan_result(plans[first_key], "hit_sl")
    plans[first_key]["closed_at"] = datetime.now(timezone.utc).isoformat()
    _save_active_plans(plans, config)
    setup.trigger_level = different_trigger
    setup.locked_sig = ""
    assert record_active_plan(analysis, setup, config) is True
    assert len(load_active_plans(config)) == 2


def test_closed_trade_is_not_reoffered_as_a_new_recommendation(tmp_path):
    """After a trade closes, its exact trigger/day is not re-announced."""
    analysis = _synthetic_analysis(tmp_path)
    setup = analysis.setup
    assert setup is not None
    config = Config()
    config.out_dir = tmp_path
    assert record_active_plan(analysis, setup, config) is True

    plans = load_active_plans(config)
    key = next(iter(plans))
    set_plan_result(plans[key], "hit_sl")
    plans[key]["closed_at"] = datetime.now(timezone.utc).isoformat()
    _save_active_plans(plans, config)

    setup.locked_sig = ""
    assert apply_active_plans([analysis], config) == 0
    assert analysis.setup is setup and analysis.setup.locked_sig == ""
    assert suppress_duplicate_setups([analysis], config) == 1
    assert analysis.setup is None


def test_every_recommendation_is_added_to_tracking(tmp_path):
    from botv2.cli import _log_recommendations

    analysis = _synthetic_analysis(tmp_path)
    config = Config()
    config.out_dir = tmp_path
    assert analysis.setup is not None

    assert _log_recommendations([analysis], config) == 1
    plans = load_active_plans(config)
    assert len(plans) == 1
    plan = next(iter(plans.values()))
    assert plan["entry"] == pytest.approx(analysis.setup.entry)
    assert plan["sig"] in (tmp_path / "recommended_trades.md").read_text(encoding="utf-8")


def test_plan_result_pnl_points_and_r():
    short = {"direction": -1, "entry": 100.0, "sl": 101.0, "tp": 98.0, "risk_usd": 300.0}
    set_plan_result(short, "hit_sl")
    assert short["pnl_points"] == -1.0
    assert short["pnl_r"] == -1.0
    assert short["pnl_usd"] == -300.0
    set_plan_result(short, "hit_tp")
    assert short["pnl_points"] == 2.0
    assert short["pnl_r"] == 2.0
    assert short["pnl_usd"] == 600.0

    long = {"direction": 1, "entry": 100.0, "sl": 99.0, "tp": 103.0, "risk_usd": 300.0}
    set_plan_result(long, "hit_tp")
    assert long["pnl_points"] == 3.0
    assert long["pnl_r"] == 3.0
    assert long["pnl_usd"] == 900.0
    set_plan_result(long, "hit_sl")
    assert long["pnl_points"] == -1.0
    assert long["pnl_r"] == -1.0
    assert long["pnl_usd"] == -300.0


def test_tracking_kpis_compute_metrics():
    from botv2.dashboard import _tracking_kpis

    trades = [
        {"is_engine": True, "pnl": 600.0, "pnl_is_usd": True, "status": "hit_tp",
         "direction": "SHORT", "_opened_at": "2026-11-03T10:00:00+00:00"},
        {"is_engine": True, "pnl": -300.0, "pnl_is_usd": True, "status": "hit_sl",
         "direction": "SHORT", "_opened_at": "2026-11-03T11:00:00+00:00"},
        {"is_engine": True, "pnl": 300.0, "pnl_is_usd": True, "status": "hit_tp",
         "direction": "LONG", "_opened_at": "2026-11-04T10:00:00+00:00"},
        {"is_engine": True, "pnl": None, "pnl_is_usd": True, "status": "recommended",
         "direction": "LONG", "_opened_at": "2026-11-04T11:00:00+00:00"},
    ]
    html = _tracking_kpis(trades)
    assert "+$600.00" in html
    assert "3 closed · 1 open" in html
    assert "66.7%" in html
    assert "3.00" in html
    assert '<span class="pos">$450</span><span class="sep">/</span><span class="neg">$300</span>' in html
    assert "50.0%" in html
    assert "days traded" in html


def test_cet_time_is_europe_berlin():
    assert _cet_time("2026-11-02T18:26:47+00:00") == "2026-11-02 19:26 CET"
