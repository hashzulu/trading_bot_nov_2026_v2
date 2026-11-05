"""Strategy engine: multi-timeframe context, confluence scoring, trade plans,
and the pre-trade checklist distilled from the journal."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

from .config import Config
from .data import SymbolData, tf_label
from .features import (
    Divergence,
    Event,
    HSPattern,
    Structure,
    Sweep,
    atr,
    bos_events,
    bos_reference,
    build_structure,
    choch_events,
    detect_divergences,
    detect_head_shoulders,
    detect_sweeps,
    major_swings,
    ny_open_window,
    real_swings,
    detect_ranges,
    range_breakout_events,
    range_events_from_boxes,
    rolling_atr,
    rsi,
    session_label,
    sweep_reversal_events,
    unswept_indices,
)

LIVE = object()  # sentinel: use the last bar (live chart), not a historical bar


# --------------------------------------------------------------------------- #
# Views
# --------------------------------------------------------------------------- #

@dataclass
class TFView:
    tf: int
    frame: pd.DataFrame
    rsi: pd.Series
    atr: pd.Series
    structure: Structure
    sweeps: list[Sweep]
    divergences: list[Divergence]
    trend: int
    hs: list[HSPattern] = field(default_factory=list)
    major_swings: list[Swing] = field(default_factory=list)
    bos_events: list[Event] = field(default_factory=list)
    choch_events: list[Event] = field(default_factory=list)
    range_breakouts: list[Event] = field(default_factory=list)
    sweep_reversals: list[Event] = field(default_factory=list)
    ranges: list[object] = field(default_factory=list)
    range_source: pd.DataFrame | None = None
    unswept: set[int] = field(default_factory=set)

    @property
    def last_idx(self) -> int:
        return len(self.frame) - 1

    @property
    def price(self) -> float:
        return float(self.frame["close"].iloc[-1])

    @property
    def atr_now(self) -> float:
        value = float(self.atr.iloc[-1])
        return value if np.isfinite(value) and value > 0 else float(self.frame["close"].std() or 1.0)

    @property
    def rsi_now(self) -> float:
        return float(self.rsi.iloc[-1])

    def window(self, bars: int) -> pd.DataFrame:
        return self.frame.iloc[-bars:]

    def trigger_events(self) -> list[Event]:
        """1m uses the v2 LH/HL BOS plus CHoCH; other frames use structure events."""
        if self.tf in (1, 5):
            events = self.bos_events + self.choch_events
            return sorted(events, key=lambda e: e.idx)
        return self.structure.events

    def events_in_window(self, bars: int) -> list[Event]:
        start = self.last_idx - bars + 1
        return [e for e in self.trigger_events() if e.idx >= start]

    def last_trigger_event(self) -> Event | None:
        events = self.trigger_events()
        return events[-1] if events else None

    def sweeps_in_window(self, bars: int) -> list[Sweep]:
        start = self.last_idx - bars + 1
        return [s for s in self.sweeps if s.idx >= start]

    def divergences_in_window(self, bars: int) -> list[Divergence]:
        start = self.last_idx - bars + 1
        return [d for d in self.divergences if d.idx >= start]

    def swing_source(self) -> list[Swing]:
        """Real/major swings when available, otherwise the fractal structure swings."""
        return self.major_swings or self.structure.swings

    def last_swing(self, kind: str) -> Swing | None:
        for swing in reversed(self.swing_source()):
            if swing.kind == kind:
                return swing
        return None

    def last_unswept_swing(self, kind: str) -> Swing | None:
        for swing in reversed(self.swing_source()):
            if swing.kind == kind and swing.idx in self.unswept:
                return swing
        return None


def build_view(frame: pd.DataFrame, tf: int, config: Config) -> TFView:
    rsi_series = rsi(frame["close"], config.rsi_period)
    atr_series = atr(frame, config.atr_period)
    structure = build_structure(frame, config.swing_k)
    real_params = config.real_swing_params.get(tf)
    if real_params is not None:
        majors = real_swings(frame, atr_series, **real_params)
    else:
        params = config.major_swing_params.get(tf, {})
        majors = major_swings(frame, atr_series, structure.raw_swings, **params)
    sweeps = detect_sweeps(frame, atr_series, structure, config.sweep_lookback)
    divergences = detect_divergences(frame, rsi_series, structure,
                                     config.divergence_min_bars)
    hs = detect_head_shoulders(frame, structure, atr_series)
    v2_bos = bos_events(frame, structure) if tf in (1, 5) else []
    v2_choch = choch_events(frame, structure) if tf in (1, 5) else []
    range_breaks = []
    sweep_reversals = (sweep_reversal_events(frame, structure)
                       if tf == 1 else [])
    range_params = config.range_params_by_tf.get(tf)
    if range_params is not None:
        scan = config.range_scan_bars.get(tf, 0)
        range_source = frame.tail(scan) if scan and len(frame) > scan else frame
        boxes = detect_ranges(range_source,
                              rolling_atr(range_source, config.atr_period),
                              **range_params)
    else:
        range_source = None
        boxes = []
    unswept = unswept_indices(frame, majors or structure.swings)
    return TFView(tf, frame, rsi_series, atr_series, structure, sweeps, divergences,
                  structure.trend, hs, majors, v2_bos, v2_choch,
                  range_breaks, sweep_reversals, boxes, range_source, unswept)


@dataclass(frozen=True)
class PotentialEntries:
    """The three potential entry levels shown in the TL;DR and the 1-min chart."""
    long_above: float | None = None
    short_below: float | None = None
    retest: float | None = None


def potential_entries(view: TFView, atr_v: float | None = None,
                      lookback: int = 260) -> PotentialEntries:
    """Nearest un-tapped 1-min swings plus the last broken level.

    This is the single source of truth: ``narrative.headline`` (TL;DR) and
    ``charts._entry_levels`` (1-min PNG) both call it, so the text and the
    picture can never disagree.
    """
    if view is None or len(view.frame) == 0:
        return PotentialEntries()
    price = view.price
    a_e = float(atr_v if atr_v is not None else view.atr_now) or 1.0
    start_idx = max(0, view.last_idx - lookback)

    def untapped(swing) -> bool:
        after = view.frame.iloc[swing.idx + 1:]
        if len(after) == 0:
            return False
        return (bool(after["high"].max() <= swing.price) if swing.kind == "H"
                else bool(after["low"].min() >= swing.price))

    if view.tf == 1:
        lh, _ = bos_reference(view.frame, view.structure.swings, 1)
        hl, _ = bos_reference(view.frame, view.structure.swings, -1)
        last = view.last_trigger_event()
        return PotentialEntries(
            long_above=lh if lh is not None and price < lh else None,
            short_below=hl if hl is not None and price > hl else None,
            retest=(last.level if last is not None else None),
        )

    recent = [sw for sw in view.swing_source() if sw.idx >= start_idx]
    highs = [sw for sw in recent
             if sw.kind == "H" and price < sw.price <= price + 20 * a_e and untapped(sw)]
    lows = [sw for sw in recent
            if sw.kind == "L" and price - 20 * a_e <= sw.price < price and untapped(sw)]
    return PotentialEntries(
        long_above=min((sw.price for sw in highs), default=None),
        short_below=max((sw.price for sw in lows), default=None),
        retest=(view.last_trigger_event().level if view.last_trigger_event() else None),
    )


# --------------------------------------------------------------------------- #
# Levels & positions
# --------------------------------------------------------------------------- #

@dataclass
class Level:
    price: float
    side: str                      # "resistance" / "support" / "pivot"
    tfs: tuple[int, ...]
    sources: tuple[str, ...]
    dist_atr: float = 0.0
    dist_r: float | None = None

    @property
    def label(self) -> str:
        tf_text = "/".join(tf_label(t) for t in self.tfs)
        return f"{self.price:.5f} ({self.side}, {tf_text})"

    @property
    def note(self) -> str:
        tf_text = "/".join(tf_label(t) for t in self.tfs)
        if self.side == "resistance":
            return f"{tf_text} swing high (liquidity above)"
        if self.side == "support":
            return f"{tf_text} swing low (liquidity below)"
        return f"{tf_text} pivot cluster"


_TF_WEIGHT = {1440: 4.0, 240: 3.0, 60: 2.0, 30: 1.5, 15: 1.2, 5: 1.0, 1: 0.8}


def cluster_levels(views: dict[int, TFView], ref_atr: float, price: float,
                   tol_atr: float = 0.6) -> list[Level]:
    raw: list[tuple[float, int, str]] = []
    for tf, view in views.items():
        source = view.major_swings if (tf in (240, 60, 15, 5, 1) and view.major_swings) \
            else view.structure.swings
        source = [sw for sw in source if sw.idx in view.unswept]
        for sw in source[-14:]:
            side = "resistance" if sw.price >= price else "support"
            raw.append((sw.price, tf, f"{tf_label(tf)} swing {'high' if sw.kind == 'H' else 'low'}"))
        # untouched extremes matter even if older
        for sw in [s for s in (view.major_swings
                               if tf in (240, 60, 15, 5, 1) and view.major_swings
                               else view.structure.swings)
                   if s.idx in view.unswept]:
            if sw.kind == "H" and sw.price >= price:
                raw.append((sw.price, tf, f"{tf_label(tf)} swing high"))
            if sw.kind == "L" and sw.price <= price:
                raw.append((sw.price, tf, f"{tf_label(tf)} swing low"))

    if not raw:
        return []

    raw.sort(key=lambda item: item[0])
    tol = max(ref_atr * tol_atr, 1e-12)
    clusters: list[list[tuple[float, int, str]]] = []
    for item in raw:
        if clusters and abs(item[0] - clusters[-1][-1][0]) <= tol:
            clusters[-1].append(item)
        else:
            clusters.append([item])

    levels: list[Level] = []
    for cluster in clusters:
        weights = np.array([_TF_WEIGHT.get(tf, 1.0) for _, tf, _ in cluster])
        prices = np.array([p for p, _, _ in cluster])
        rep = float(np.average(prices, weights=weights))
        tfs = tuple(sorted({tf for _, tf, _ in cluster}, reverse=True))
        sources = tuple(dict.fromkeys(src for _, _, src in cluster))
        side = "pivot" if abs(rep - price) <= tol else ("resistance" if rep > price else "support")
        level = Level(rep, side, tfs, sources)
        level.dist_atr = (rep - price) / ref_atr if ref_atr else 0.0
        levels.append(level)

    # keep the most relevant cluster per band
    keep: list[Level] = []
    for level in levels:
        if abs(level.dist_atr) <= 30:
            keep.append(level)
    return keep


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #

@dataclass
class Factor:
    name: str
    points: int
    detail: str


@dataclass
class ChecklistItem:
    key: str
    status: str          # PASS / WARN / FAIL / MANUAL
    text: str
    lesson: str = ""


@dataclass
class Setup:
    symbol: str
    direction: int                 # +1 long, -1 short
    grade: str                     # A / B / C
    score: int
    max_score: int
    entry: float
    entry_note: str
    sl: float
    tp: float
    tp_note: str
    rr: float
    risk_atr: float
    trigger_tf: int
    trigger_kind: str
    trigger_level: float
    bars_since_trigger: int
    factors: list[Factor]
    thesis: str
    invalidation: str
    checklist: list[ChecklistItem]
    levels: list[Level]
    session: str
    rules_ticked: list[str] = field(default_factory=list)
    rules_missing: list[str] = field(default_factory=list)
    rule_count: int = 0
    # set when the plan is pinned to an earlier recommendation (levels must not move)
    locked_sig: str = ""
    # when the recommendation was logged (CET display uses this, not the clock)
    recommended_at: datetime | None = None

    @property
    def direction_label(self) -> str:
        return "LONG" if self.direction > 0 else "SHORT"

    @property
    def risk_distance(self) -> float:
        return abs(self.entry - self.sl)

    def position_size(self, risk_usd: float) -> float:
        return risk_usd / self.risk_distance if self.risk_distance else 0.0

    @property
    def signature(self) -> str:
        return f"{self.symbol}|{self.direction_label}|{self.grade}|{self.trigger_tf}|{round(self.entry, 5)}"


@dataclass
class WatchItem:
    direction: int
    missing: list[str]
    text: str


@dataclass
class SymbolAnalysis:
    symbol: str
    views: dict[int, TFView]
    bias: int                      # +1 / -1 / 0
    bias_label: str
    bias_reason: str
    levels: list[Level]
    setup: Setup | None
    watch: WatchItem | None
    notes: list[str] = field(default_factory=list)
    price: float = 0.0
    session: str = ""
    atr_entry: float = 0.0
    entry_tf: int = 5
    journal: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Bias
# --------------------------------------------------------------------------- #

def determine_bias(views: dict[int, TFView], config: Config) -> tuple[int, str, str]:
    htf = views.get(240) or views.get(60)
    mtf = views.get(60) or views.get(15)
    if htf is None:
        return 0, "N/A", "insufficient data for a higher-timeframe read"

    h = htf.trend
    m = mtf.trend if mtf is not None else 0
    h_rsi = htf.rsi_now
    m_rsi = mtf.rsi_now if mtf is not None else 50.0

    # a flat higher timeframe falls back to the intraday read
    if h == 0 and m != 0:
        return (m, "LONG (soft)" if m > 0 else "SHORT (soft)",
                f"{tf_label(htf.tf)} is flat; {tf_label(mtf.tf)} structure is "
                f"{'bullish' if m > 0 else 'bearish'}")
    if m == 0 and h != 0:
        return (h, "LONG (soft)" if h > 0 else "SHORT (soft)",
                f"{tf_label(htf.tf)} structure is {'bullish' if h > 0 else 'bearish'}; "
                f"intraday is flat")

    if h == 1 and m == 1:
        return 1, "LONG (trend)", f"{tf_label(htf.tf)} structure bullish, {tf_label(mtf.tf)} aligned bullish"
    if h == -1 and m == -1:
        return -1, "SHORT (trend)", f"{tf_label(htf.tf)} structure bearish, {tf_label(mtf.tf)} aligned bearish"
    if h == 1 and m == -1:
        if m_rsi <= 45:
            return 1, "LONG (pullback)", (
                f"{tf_label(htf.tf)} still bullish while {tf_label(mtf.tf)} pulls back "
                f"(RSI {m_rsi:.1f}) — continuation long context"
            )
        return 0, "NEUTRAL (conflict)", "higher timeframe bullish but intraday sell-off is not oversold"
    if h == -1 and m == 1:
        if m_rsi >= 55:
            return -1, "SHORT (pullback)", (
                f"{tf_label(htf.tf)} still bearish while {tf_label(mtf.tf)} bounces "
                f"(RSI {m_rsi:.1f}) — continuation short context"
            )
        return 0, "NEUTRAL (conflict)", "higher timeframe bearish but intraday bounce is not overbought"
    if h == 0 and m == 0:
        return 0, "NEUTRAL (range)", "no clean structure on either context timeframe"
    return (h, "LONG (soft)" if h > 0 else "SHORT (soft)",
            f"{tf_label(htf.tf)} {'bullish' if h > 0 else 'bearish'} with mixed intraday structure")


# --------------------------------------------------------------------------- #
# Setup detection
# --------------------------------------------------------------------------- #

def _htf_sweep(views: dict[int, TFView], direction: int, bars: int) -> tuple[bool, float | None]:
    """Did the entry timeframe wick through the 1h swing and reclaim it?

    4H is only used as a fallback when the 1H view is unavailable.
    """
    entry = views.get(min(views)) if views else None
    if entry is None:
        return False, None
    tf = 60 if 60 in views else 240
    view = views.get(tf)
    if view is None:
        return False, None
    swing = view.last_unswept_swing("L" if direction > 0 else "H")
    if swing is None:
        return False, None
    window = entry.window(bars)
    if direction > 0:
        touched = (window["low"] <= swing.price) & (window["close"] > swing.price)
    else:
        touched = (window["high"] >= swing.price) & (window["close"] < swing.price)
    if bool(touched.any()):
        return True, swing.price
    return False, None


def _swing_respected(view: TFView, direction: int, bars: int,
                     tol_atr: float = 0.35) -> tuple[bool, float | None]:
    """Price wicked into a swing level and closed back on the right side (no break)."""
    swing = view.last_unswept_swing("L" if direction > 0 else "H")
    if swing is None:
        return False, None
    window = view.window(bars)
    atr_v = view.atr_now
    tol = tol_atr * atr_v
    if direction > 0:
        touched = (window["low"] <= swing.price + tol) & (window["close"] > swing.price)
    else:
        touched = (window["high"] >= swing.price - tol) & (window["close"] < swing.price)
    return bool(touched.any()), swing.price


def _rsi_confirm(view: TFView, direction: int) -> bool:
    """5-min RSI confirmation, which must directly follow a divergence.

    Price makes a HL/LH *and* RSI makes a HL/LH, with the immediately preceding
    swing carrying a same-direction RSI divergence. In isolation (no divergence
    at the prior swing) this does not count.
    """
    kind = "L" if direction > 0 else "H"
    swings = [s for s in view.structure.swings if s.kind == kind]
    if len(swings) < 2:
        return False
    a, b = swings[-2], swings[-1]
    rsi_a = float(view.rsi.iloc[a.idx])
    rsi_b = float(view.rsi.iloc[b.idx])
    if not (rsi_a == rsi_a and rsi_b == rsi_b):
        return False

    if direction > 0:
        confirmed = b.price > a.price and rsi_b > rsi_a
    else:
        confirmed = b.price < a.price and rsi_b < rsi_a
    if not confirmed:
        return False

    # The divergence must end at the prior swing (a), making this confirmation
    # the immediate next swing rather than an isolated HL/LH.
    return any(d.direction == direction and d.idx == a.confirm_idx
               for d in view.divergences)


def _pick_target(levels: list[Level], entry: float, risk: float, direction: int,
                 config: Config) -> tuple[float, str]:
    candidates: list[tuple[float, str]] = []
    for level in levels:
        if direction > 0 and level.price > entry:
            dist = level.price - entry
            if 1.0 * risk <= dist <= 8 * risk:
                candidates.append((level.price, level.note))
        if direction < 0 and level.price < entry:
            dist = entry - level.price
            if 1.0 * risk <= dist <= 8 * risk:
                candidates.append((level.price, level.note))
    if candidates:
        candidates.sort(key=lambda c: abs(c[0] - entry))
        return candidates[0]
    target = entry + direction * risk * max(config.target_rr, 1.5)
    return target, f"{max(config.target_rr, 1.5):.1f}R measured move"


def _checklist(setup: Setup, views: dict[int, TFView], config: Config,
               trigger_confirmed: bool, extended: bool) -> list[ChecklistItem]:
    items: list[ChecklistItem] = []

    pattern_ok = ("H&S" in setup.trigger_kind
                  or setup.trigger_kind in ("BOS", "CHoCH", "Range breakout",
                                            "Range sweep", "HL sweep", "LH sweep"))
    items.append(ChecklistItem(
        "pattern_complete",
        "PASS" if pattern_ok else "WARN",
        f"Trigger is a completed {setup.trigger_kind}." if pattern_ok
        else "Pattern is not a completed break — wait for the neck/cluster to close through.",
        "2026-09-02: entered before the right shoulder existed; -$1,029.",
    ))
    items.append(ChecklistItem(
        "stop_discipline", "MANUAL",
        "Never widen the stop after entry — take the loss at the planned level.",
        "Journal: widening the stop is the worst leak (-$7,127 over 53 trades, 17% win).",
    ))
    items.append(ChecklistItem(
        "daily_guard", "MANUAL",
        f"Max {config.max_trades_per_day}/day, stop after "
        f"{config.stop_after_consecutive_losses} consecutive losses, one trade at a time.",
        "Account rules; 3rd+ trade of the day has been your weakest sequence.",
    ))
    from .journal import TOP_RULES
    items.append(ChecklistItem(
        "top_rules",
        "PASS" if setup.rule_count >= config.min_top_rules else "FAIL",
        f"{setup.rule_count}/{len(TOP_RULES)} top rules ticked "
        f"({', '.join(TOP_RULES.get(k, k) for k in setup.rules_ticked) or 'none'}); "
        f"trigger: {setup.trigger_kind}.",
        "Journal: >=2 rules = 58% win / PF 1.95; 0-1 rules is a losing cohort.",
    ))
    items.append(ChecklistItem(
        "structure_confirmed",
        "PASS" if trigger_confirmed else "FAIL",
        "Trigger is a completed structure break." if trigger_confirmed
        else "Trigger is still forming — wait for the break to close.",
        "2026-09-02: entered an incomplete H&S and lost $1,029.",
    ))

    has_div = any(f.name.startswith("RSI divergence") for f in setup.factors)
    # aligned with the higher-timeframe structure => continuation, divergence not required
    continuation = any(f.name.startswith("HTF alignment") for f in setup.factors)
    if has_div:
        status, text = "PASS", "Divergence present."
    elif continuation:
        status, text = "PASS", ("Continuation with the HTF trend — divergence not required "
                                "(a sweep + structure break is the confirmation).")
    else:
        status, text = "WARN", "No divergence: entering a reversal without momentum confirmation."
    items.append(ChecklistItem("divergence", status, text,
                               "2026-08-12: took a reversal short with no divergence and lost."))

    items.append(ChecklistItem(
        "entry_not_extended",
        "PASS" if not extended else "PASS",
        "Price ran past the trigger, so the plan uses the retest limit — do not chase."
        if extended else "Entry is close to the trigger level.",
        "2026-07-29: entered after the neck broke, 1-min RSI was already elevated.",
    ))

    items.append(ChecklistItem(
        "stop_size",
        "PASS" if setup.risk_atr <= config.max_sl_atr else "WARN",
        f"Stop is {setup.risk_atr:.2f} ATR ({setup.risk_distance:.5f}).",
        "2026-10-06: widened the stop from 1R to 2.88R and lost $262.",
    ))

    items.append(ChecklistItem(
        "rr_ok",
        "PASS" if setup.rr >= config.min_rr else "FAIL",
        f"R:R = 1:{setup.rr:.2f}.",
        "Several losses came from sub-1R targets or closed early.",
    ))

    aligned = any(f.name.startswith("HTF alignment") for f in setup.factors)
    pullback = "pullback" in setup.thesis.lower() or "pullback" in setup.trigger_kind.lower()
    items.append(ChecklistItem(
        "htf_aligned",
        "PASS" if (aligned or pullback) else "WARN",
        "Aligned with higher-timeframe structure." if aligned
        else ("Continuation in the HTF trend, entering the intraday pullback."
              if pullback else "Counter-trend relative to the 4h/1h structure."),
        "Losses clustered when trading against the 1-hour read.",
    ))

    items.append(ChecklistItem(
        "news_window", "MANUAL",
        f"Check the calendar for tier-1 news within ±{config.news_guard_minutes} min.",
        "2026-08-12: entered 5 minutes before news and got stopped out.",
    ))
    return items


def detect_setup(symbol: str, views: dict[int, TFView], bias: int, bias_label: str,
                 levels: list[Level], config: Config,
                 entry_tf: int | None = None) -> tuple[Setup | None, WatchItem | None]:
    if entry_tf is None:
        entry_tf = config.entry_tf if config.entry_tf in views else min(views)
    entry = views[entry_tf]
    price = entry.price
    atr_e = entry.atr_now
    window_bars = config.trigger_window
    floor = config.rsi_floor_for(symbol)
    ceiling = config.rsi_ceiling_for(symbol)

    if bias == 0:
        return None, None

    direction = bias
    # The journal's momentum rules are 5-min rules; the trigger is the entry timeframe.
    signal_view = views.get(5) or entry
    window_start = entry.last_idx - window_bars + 1
    events = [e for e in entry.events_in_window(window_bars) if e.direction == direction]
    range_breaks = [e for e in entry.range_breakouts
                    if e.direction == direction and e.idx >= window_start]
    sweep_reversals = [e for e in entry.sweep_reversals
                       if e.direction == direction and e.idx >= window_start]
    sweeps = [s for s in entry.sweeps_in_window(window_bars) if s.direction == direction]
    range_sweeps = [s for s in sweeps if s.source == "range"]
    divs = [d for d in signal_view.divergences_in_window(window_bars) if d.direction == direction]
    rsi_window = signal_view.rsi.iloc[-window_bars:]
    rsi_extreme = float(rsi_window.min()) if direction > 0 else float(rsi_window.max())
    rsi_last = float(signal_view.rsi.iloc[-1])
    htf_swept, htf_level = _htf_sweep(views, direction, window_bars)

    # 1-minute patterns: a completed H&S neckline break is a first-class trigger.
    hs_matches = [h for h in entry.hs if h.direction == direction]
    hs_complete = [h for h in hs_matches if h.complete
                   and (entry.last_idx - (h.break_idx or 0)) <= window_bars]
    hs_forming = [h for h in hs_matches if not h.complete]
    hs_trigger = hs_complete[-1] if hs_complete else None

    factors: list[Factor] = []
    score = 0

    if direction > 0:
        if rsi_extreme <= floor:
            deep = rsi_extreme <= config.rsi_deep_floor
            pts = config.pts_rsi_deep if deep else config.pts_rsi_extreme
            factors.append(Factor(f"{tf_label(signal_view.tf)} RSI floor", pts,
                                  f"RSI low of {rsi_extreme:.1f} in the last {window_bars} bars "
                                  f"(floor {floor:.0f})"))
            score += pts
    else:
        if rsi_extreme >= ceiling:
            deep = rsi_extreme >= config.rsi_deep_ceiling
            pts = config.pts_rsi_deep if deep else config.pts_rsi_extreme
            factors.append(Factor(f"{tf_label(signal_view.tf)} RSI ceiling", pts,
                                  f"RSI high of {rsi_extreme:.1f} in the last {window_bars} bars "
                                  f"(ceiling {ceiling:.0f})"))
            score += pts

    if divs:
        d = divs[-1]
        factors.append(Factor(f"RSI divergence ({d.label})", config.pts_divergence,
                              f"swing {d.p1:.5f} → {d.p2:.5f} with RSI {d.rsi1:.1f} → {d.rsi2:.1f}"))
        score += config.pts_divergence

    # Trigger selection: H&S neckline break has priority, otherwise the most
    # recent of BOS/CHoCH, range breakout, range sweep, or HL/LH sweep.
    trigger_kind = trigger_level = None
    trigger_idx = None
    trigger_price = price
    if hs_trigger is not None:
        trigger_kind = "H&S neckline break"
        trigger_level = hs_trigger.neckline
        trigger_idx = hs_trigger.break_idx
        trigger_price = float(entry.frame["close"].iloc[trigger_idx])
        factors.append(Factor(f"1-min {hs_trigger.label}", config.pts_hs,
                              f"shoulders {hs_trigger.left_price:.5f}/{hs_trigger.right_price:.5f}, "
                              f"head {hs_trigger.head_price:.5f}, neckline "
                              f"{hs_trigger.neckline:.5f} broken "
                              f"{entry.last_idx - trigger_idx} bars ago"))
        score += config.pts_hs
    else:
        candidates = []
        for event in events:
            pts = config.pts_choch if event.kind == "CHoCH" else config.pts_bos
            detail = f"close {event.price:.5f} through {event.level:.5f}"
            if event.label and event.kind == "BOS":
                detail = (f"close {'above' if direction > 0 else 'below'} "
                          f"{event.label} {event.level:.5f} ({event.price:.5f})")
            elif event.label:
                detail = (f"close {event.price:.5f} through {event.level:.5f} "
                          f"after a 1m {event.label} reversal")
            candidates.append((event.idx, event.kind, event.level, event.price,
                               pts, detail))
        for event in range_breaks:
            candidates.append((event.idx, "Range breakout", event.level,
                               event.price, config.pts_bos,
                               f"close {event.price:.5f} beyond {event.level:.5f}"))
        for sweep in range_sweeps:
            candidates.append((sweep.idx, "Range sweep", sweep.level,
                               float(entry.frame["close"].iloc[sweep.idx]),
                               config.pts_range_sweep,
                               f"{sweep.level:.5f} swept by {sweep.wick:.5f} "
                               f"and reclaimed"))
        for event in sweep_reversals:
            candidates.append((event.idx, event.kind, event.level, event.price,
                               config.pts_sweep,
                               f"swept {event.swept:.5f} and closed "
                               f"{'above' if direction > 0 else 'below'} "
                               f"{event.level:.5f} ({event.price:.5f})"))
        if candidates:
            _, trigger_kind, trigger_level, trigger_price, pts, detail = max(
                candidates, key=lambda item: item[0])
            trigger_idx = max(candidates, key=lambda item: item[0])[0]
            factors.append(Factor(
                f"{trigger_kind} {'bullish' if direction > 0 else 'bearish'} "
                f"({tf_label(entry_tf)})", pts, detail))
            score += pts

    if sweeps:
        sweep = sweeps[-1]
        factors.append(Factor(f"Liquidity sweep of {sweep.source} level", config.pts_sweep,
                              f"{sweep.level:.5f} swept by {sweep.wick:.5f} and reclaimed"))
        score += config.pts_sweep
        if sweep.source == "range":
            factors.append(Factor("Range sweep (journal edge)", config.pts_range_sweep,
                                  "the swept level was a range boundary, not a single swing"))
            score += config.pts_range_sweep

    if htf_swept and htf_level is not None:
        factors.append(Factor("HTF swing reclaimed", config.pts_htf_swing,
                              f"higher-timeframe level {htf_level:.5f} swept and reclaimed"))
        score += config.pts_htf_swing

    aligned = bias_label.startswith(("LONG (trend)", "SHORT (trend)"))
    if aligned:
        factors.append(Factor("HTF alignment", config.pts_trend_alignment, bias_label))
        score += config.pts_trend_alignment

    last_ts = entry.frame.index[-1]
    if ny_open_window(last_ts):
        factors.append(Factor("NY open window", config.pts_session, "13:00-15:30 UTC impulse window"))
        score += config.pts_session

    max_score = (config.pts_rsi_deep + config.pts_divergence + config.pts_hs
                 + config.pts_sweep + config.pts_range_sweep + config.pts_htf_swing
                 + config.pts_trend_alignment + config.pts_session)

    have_trigger = trigger_level is not None
    rsi_ok = (rsi_extreme <= floor) if direction > 0 else (rsi_extreme >= ceiling)
    rsi_confirm = bool(rsi_ok and (
        (rsi_last - rsi_extreme >= config.rsi_confirm_points) if direction > 0
        else (rsi_extreme - rsi_last >= config.rsi_confirm_points)
    ))
    have_context = bool(divs) or bool(sweeps) or rsi_ok or hs_trigger is not None

    late = False
    if have_trigger and trigger_level is not None:
        chase = abs(price - trigger_level) / atr_e if atr_e else 0.0
        late = chase > config.max_trigger_chase_atr

    # ----- pre-trade rule gate (journal: >=2 top rules -> 58% win / PF 1.95) -----
    five_min = views.get(5) or entry
    htf_respected, htf_respect_level = _swing_respected(
        views.get(60) or views.get(240) or entry, direction, window_bars)
    min5_swept = bool([s for s in five_min.sweeps_in_window(window_bars)
                       if s.direction == direction])
    min5_respected, min5_level = _swing_respected(five_min, direction, window_bars)
    rules: dict[str, bool] = {
        "sweptSwing": bool(htf_swept or htf_respected),
        "min5RespectedSwept": bool(min5_swept or min5_respected),
        "watchingRsi": bool(rsi_ok),
        "rsiDivergence": bool(divs),
        "rsiConfirm": _rsi_confirm(five_min, direction),
    }
    rules_ticked = [key for key, ok in rules.items() if ok]
    rules_missing = [key for key, ok in rules.items() if not ok]
    rule_count = len(rules_ticked)
    rules_ok = rule_count >= config.min_top_rules
    if (not have_trigger or not have_context or score < config.min_score_b or late
            or not rules_ok):
        missing: list[str] = []
        long_side = direction > 0
        digits = 2 if price >= 10 else 5

        def px(value: float) -> str:
            return f"{value:.{digits}f}"

        # --- structure break: the v2 1m BOS reference is an LH / HL ---
        if not have_trigger:
            if entry_tf == 1:
                level, ref = bos_reference(entry.frame, entry.structure.swings, direction)
                reversal = ("bottom→high→higher-low" if long_side
                            else "top→low→lower-high")
                side_word = "bullish" if long_side else "bearish"
                if level is not None:
                    missing.append(
                        f"1m triggers: close {'above' if long_side else 'below'} "
                        f"the last {ref} at {px(level)} "
                        f"(BOS{'↑' if long_side else '↓'} {ref}) or {side_word} "
                        f"CHoCH / range breakout / range sweep / "
                        f"{'HL' if long_side else 'LH'} sweep")
                else:
                    missing.append(
                        f"1m triggers: {side_word} BOS / CHoCH / range breakout / "
                        f"range sweep / {'HL' if long_side else 'LH'} sweep — none yet")
            else:
                swing = (entry.structure.last_swing_high if long_side
                         else entry.structure.last_swing_low)
                if swing is not None:
                    side = "high" if long_side else "low"
                    missing.append(
                        f"{tf_label(entry_tf)} structure break: a close "
                        f"{'above' if long_side else 'below'} the last {tf_label(entry_tf)} "
                        f"swing {side} at {px(swing.price)}")
                else:
                    missing.append(f"{tf_label(entry_tf)} structure break (no swing levels yet)")

        if hs_forming:
            forming = hs_forming[-1]
            missing.append(
                f"wait for the {tf_label(entry_tf)} {forming.label} to complete: a close "
                f"{'above' if long_side else 'below'} the neckline at {px(forming.neckline)} "
                f"(right shoulder formed, neckline not broken)")

        # --- RSI zone: show the level, current value and window extreme ---
        if not rsi_ok:
            if long_side:
                missing.append(
                    f"{tf_label(signal_view.tf)} RSI into the {floor:.0f} floor "
                    f"(now {rsi_last:.1f}, window low {rsi_extreme:.1f})")
            else:
                missing.append(
                    f"{tf_label(signal_view.tf)} RSI into the {ceiling:.0f} ceiling "
                    f"(now {rsi_last:.1f}, window high {rsi_extreme:.1f})")

        # --- divergence: name the prior swing + its RSI ---
        if not divs:
            reference = [sw for sw in signal_view.structure.swings
                         if sw.kind == ("L" if long_side else "H")][-1:]
            if reference:
                ref = reference[0]
                ref_rsi = float(signal_view.rsi.iloc[ref.idx])
                kind = "low" if long_side else "high"
                condition = ("a lower low on price with a higher RSI" if long_side
                             else "a higher high on price with a lower RSI")
                missing.append(
                    f"{tf_label(signal_view.tf)} RSI divergence vs the prior swing {kind} "
                    f"at {px(ref.price)} (RSI {ref_rsi:.1f} there): need {condition}")
            else:
                missing.append(f"{tf_label(signal_view.tf)} RSI divergence "
                               f"(no prior swing {'low' if long_side else 'high'} yet)")

        # --- sweep: name the swing level and, if present, the range boundaries ---
        if not sweeps:
            swing = (entry.structure.last_swing_low if long_side
                     else entry.structure.last_swing_high)
            bits: list[str] = []
            if swing is not None:
                kind = "low" if long_side else "high"
                bits.append(f"the prior {tf_label(entry_tf)} swing {kind} at {px(swing.price)}")
            window = entry.frame.iloc[-26:-1] if len(entry.frame) > 26 else entry.frame
            if len(window) >= 10:
                rng_low = float(window["low"].min())
                rng_high = float(window["high"].max())
                if (rng_high - rng_low) <= 4 * entry.atr_now:
                    side = "low" if long_side else "high"
                    level = rng_low if long_side else rng_high
                    bits.append(f"the range {side} at {px(level)} "
                                f"(range {px(rng_low)}–{px(rng_high)})")
            missing.append("a sweep of " + " or ".join(bits) if bits
                           else "a sweep of the prior swing / range liquidity")

        # --- HTF swing: name the 1h/4h level to work against ---
        if not htf_swept:
            htf_view = views.get(60) or views.get(240)
            level = None
            htf_name = "1h"
            if htf_view is not None:
                htf_name = tf_label(htf_view.tf)
                source = [s for s in (htf_view.major_swings or htf_view.structure.swings)
                          if s.idx in htf_view.unswept]
                # for a short we need a high *above* price to sweep; long the opposite
                if long_side:
                    candidates = [sw.price for sw in source
                                  if sw.kind == "L" and 0 < price - sw.price <= 20 * atr_e]
                    level = max(candidates) if candidates else None
                else:
                    candidates = [sw.price for sw in source
                                  if sw.kind == "H" and 0 < sw.price - price <= 20 * atr_e]
                    level = min(candidates) if candidates else None
            kind = "low" if long_side else "high"
            missing.append(
                f"a {htf_name} swing {kind} sweep at {px(level)} to work against"
                if level is not None else
                f"a {htf_name} swing {kind} sweep to work against")

        if late and trigger_level is not None:
            missing.append(
                f"price is {abs(price - trigger_level) / atr_e:.1f} ATR past the trigger "
                f"({px(trigger_level)}) — the break is gone, wait for a retest or a new trigger")

        if not rules_ok:
            from .journal import TOP_RULES
            labels = [TOP_RULES.get(k, k) for k in rules_missing]
            missing.append(
                f">= {config.min_top_rules} pre-trade rules ticked (have {rule_count}: "
                f"{', '.join(rules_ticked) or 'none'}); still missing: {', '.join(labels)}")
        if score < config.min_score_b:
            missing.append(f"confluence score ≥ {config.min_score_b} (now {score})")
        watch = WatchItem(
            direction,
            missing,
            f"{'Long' if direction > 0 else 'Short'} bias without a tradeable trigger. "
            f"Wait for: " + "; ".join(missing) + ".",
        )
        return None, watch

    # ----- trade plan ------------------------------------------------------ #
    assert trigger_level is not None
    extended = abs(price - trigger_level) > 0.75 * atr_e
    entry_price = trigger_level + direction * 0.15 * atr_e if extended else price
    entry_note = (
        f"rest on retest of {trigger_level:.5f}" if extended
        else "market at close, trigger fresh"
    )

    if direction > 0:
        anchors = [trigger_level]
        if sweeps:
            anchors.append(sweeps[-1].level)
        anchors.append(float(entry.window(4)["low"].min()))
        sl = min(anchors) - 0.35 * atr_e
    else:
        anchors = [trigger_level]
        if sweeps:
            anchors.append(sweeps[-1].level)
        anchors.append(float(entry.window(4)["high"].max()))
        sl = max(anchors) + 0.35 * atr_e

    risk = abs(entry_price - sl)
    if risk <= 0:
        return None, None

    tp, tp_note = _pick_target(levels, entry_price, risk, direction, config)
    rr = abs(tp - entry_price) / risk
    risk_atr = risk / atr_e

    if rr < config.min_rr:
        tp = entry_price + direction * risk * config.min_rr
        tp_note = f"{config.min_rr:.1f}R (no clean level within reach)"
        rr = config.min_rr

    setup = Setup(
        symbol=symbol,
        direction=direction,
        grade="B",                     # provisional until the checklist is scored
        score=score,
        max_score=max_score,
        entry=entry_price,
        entry_note=entry_note,
        sl=sl,
        tp=tp,
        tp_note=tp_note,
        rr=rr,
        risk_atr=risk_atr,
        trigger_tf=entry_tf,
        trigger_kind=trigger_kind or "structure break",
        trigger_level=trigger_level,
        bars_since_trigger=max(0, entry.last_idx - trigger_idx) if trigger_idx is not None else 0,
        factors=factors,
        thesis="",
        invalidation="",
        checklist=[],
        levels=levels,
        session=session_label(last_ts),
        rules_ticked=rules_ticked,
        rules_missing=rules_missing,
        rule_count=rule_count,
    )
    setup.checklist = _checklist(setup, views, config, True, extended)

    warns = sum(1 for item in setup.checklist if item.status == "WARN")
    fails = sum(1 for item in setup.checklist if item.status == "FAIL")
    if fails:
        setup.grade = "C"
    elif score >= config.min_score_a and warns <= 1 and rule_count >= 3:
        setup.grade = "A"
    elif score >= config.min_score_b:
        setup.grade = "B"
    else:
        setup.grade = "C"
    return setup, None


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

def journal_snapshot(symbol: str, views: dict[int, TFView], bias: int, bias_label: str,
                     levels: list[Level], config: Config, entry_tf: int) -> dict:
    """Everything the journal-style WATCHING / THESIS / NOTES sections need."""
    entry = views[entry_tf]
    signal_view = views.get(5) or entry
    window = config.trigger_window
    floor = config.rsi_floor_for(symbol)
    ceiling = config.rsi_ceiling_for(symbol)
    price = entry.price
    direction = bias or 0

    rsi_window = signal_view.rsi.iloc[-window:]
    rsi_min = float(rsi_window.min())
    rsi_max = float(rsi_window.max())
    rsi_now = float(signal_view.rsi.iloc[-1])

    divs = [d for d in signal_view.divergences_in_window(window)
            if direction == 0 or d.direction == direction]
    htf_swept, htf_level = _htf_sweep(views, direction, window) if direction else (False, None)
    htf_respected, htf_respect_level = ((_swing_respected(
        views.get(60) or views.get(240) or entry, direction, window))
        if direction else (False, None))
    five = views.get(5) or entry
    sweeps_1m = entry.sweeps_in_window(window)
    m5_swept = [s for s in five.sweeps_in_window(window)
                if not direction or s.direction == direction]
    m5_respect, m5_respect_level = _swing_respected(five, direction or 1, window)
    hs_all = [h for h in entry.hs if not direction or h.direction == direction]
    hs_complete = [h for h in hs_all if h.complete]
    hs_forming = [h for h in hs_all if not h.complete]
    events = [e for e in entry.events_in_window(window)
              if not direction or e.direction == direction]

    rsi_ok = ((rsi_min <= floor) if direction > 0 else (rsi_max >= ceiling)) if direction else False
    confirm = _rsi_confirm(five, direction) if direction else False
    rules = {
        "sweptSwing": bool(htf_swept or htf_respected),
        "min5RespectedSwept": bool(m5_swept or m5_respect),
        "watchingRsi": bool(rsi_ok),
        "rsiDivergence": bool(divs),
        "rsiConfirm": confirm,
    }

    def nearest_levels(tf: int, role: str, count: int = 2) -> list[float]:
        view = views.get(tf)
        if view is None:
            return []
        majors = [sw for sw in (view.major_swings or view.structure.swings)
                  if sw.idx in view.unswept]
        if role == "support":
            pool = ([sw for sw in majors if sw.kind == "L" and sw.price < price]
                    if direction > 0 else
                    [sw for sw in majors if sw.kind == "H" and sw.price > price])
        else:
            pool = ([sw for sw in majors if sw.kind == "H" and sw.price > price]
                    if direction > 0 else
                    [sw for sw in majors if sw.kind == "L" and sw.price < price])
        return [sw.price for sw in sorted(pool, key=lambda sw: abs(sw.price - price))[:count]]

    nearest_h1 = nearest_levels(60, "support")
    nearest_m5 = nearest_levels(5, "support")
    target_h1 = nearest_levels(60, "target")
    target_m5 = nearest_levels(5, "target")

    # potential entries: nearest un-tapped 1-min swings + the last broken level
    pe = potential_entries(entry)
    entry_long_above = pe.long_above
    entry_short_below = pe.short_below
    entry_retest = pe.retest

    return {
        "direction": direction,
        "bias_label": bias_label,
        "entry_tf": entry_tf,
        "price": price,
        "atr": entry.atr_now,
        "session": session_label(entry.frame.index[-1]),
        "ny_open": ny_open_window(entry.frame.index[-1]),
        "floor": floor,
        "ceiling": ceiling,
        "rsi_min": rsi_min,
        "rsi_max": rsi_max,
        "rsi_now": rsi_now,
        "rsi_timeframe": signal_view.tf,
        "divergences": [{"label": d.label} for d in divs[-2:]],
        "confirm": confirm,
        "htf_swept": htf_swept,
        "htf_level": htf_level,
        "htf_respected": htf_respected,
        "htf_respect_level": htf_respect_level,
        "m5_sweep": (None if not m5_swept else
                     {"level": m5_swept[-1].level, "source": m5_swept[-1].source}),
        "m5_respected": m5_respect,
        "m5_respect_level": m5_respect_level,
        "hs_complete": (None if not hs_complete else
                        {"label": hs_complete[-1].label,
                         "neckline": hs_complete[-1].neckline,
                         "bars": entry.last_idx - (hs_complete[-1].break_idx or 0)}),
        "hs_forming": (None if not hs_forming else
                       {"label": hs_forming[-1].label,
                        "neckline": hs_forming[-1].neckline}),
        "trigger": (None if not events else
                    {"kind": events[-1].kind, "level": events[-1].level,
                     "label": events[-1].label,
                     "bars": entry.last_idx - events[-1].idx}),
        "rules": rules,
        "rule_count": sum(1 for ok in rules.values() if ok),
        "levels_h1": nearest_h1,
        "levels_m5": nearest_m5,
        "targets_h1": target_h1,
        "targets_m5": target_m5,
        "entry_long_above": entry_long_above,
        "entry_short_below": entry_short_below,
        "entry_retest": entry_retest,
    }


def _attach_range_triggers(views: dict[int, TFView], config: Config) -> None:
    """Attach each timeframe's own boxes and resolved range events."""
    for view in views.values():
        source = view.range_source
        if source is None or not view.ranges:
            continue
        source_atr = rolling_atr(source, config.atr_period)
        breaks, sweeps = range_events_from_boxes(
            view.frame, source, source_atr, view.ranges,
            config.range_sweep_tol_atr, config.range_reclaim_bars,
            config.range_break_close_atr, config.range_resolve_max_wait)
        seen_breaks = set()
        unique_breaks = []
        for event in breaks:
            key = (event.idx, event.direction)
            if key in seen_breaks:
                continue
            seen_breaks.add(key)
            unique_breaks.append(event)
        seen_sweeps = set()
        unique_sweeps = []
        for sweep in sweeps:
            key = (sweep.idx, sweep.direction)
            if key in seen_sweeps:
                continue
            seen_sweeps.add(key)
            unique_sweeps.append(sweep)
        view.range_breakouts = unique_breaks
        view.sweeps = [s for s in view.sweeps if s.source != "range"] + unique_sweeps
        view.sweeps.sort(key=lambda s: s.idx)


def analyse_symbol(data: SymbolData, config: Config) -> SymbolAnalysis:
    wanted = {*config.context_tfs, *config.setup_tfs, config.entry_tf}
    if config.auto_entry_tf and 1 in data.frames:
        wanted.add(1)          # 1-minute triggers when the M1 export exists
    wanted = sorted(wanted, reverse=True)
    views: dict[int, TFView] = {}
    for tf in wanted:
        frame = data.df(tf)
        if frame is None or len(frame) < max(config.min_bars, 40):
            continue
        views[tf] = build_view(frame, tf, config)

    if not views:
        return SymbolAnalysis(data.symbol, {}, 0, "N/A", "no usable data", [], None, None,
                              notes=list(data.notes))

    _attach_range_triggers(views, config)
    entry_tf = config.entry_tf if config.entry_tf in views else min(views)
    if config.auto_entry_tf and 1 in views:
        entry_tf = 1
    entry = views[entry_tf]
    price = entry.price
    bias, bias_label, bias_reason = determine_bias(views, config)
    levels = cluster_levels(views, entry.atr_now, price)
    setup, watch = detect_setup(data.symbol, views, bias, bias_label, levels, config, entry_tf)

    notes = list(data.notes)
    age = data.last_bar_age_minutes()
    if age != float("inf"):
        notes.append(f"newest bar is {age:.0f} min old")
        if age > config.stale_after_minutes:
            notes.append(
                f"⚠ data looks stale (>{config.stale_after_minutes} min) — re-export from MT4 "
                f"before acting on this read")
    if config.entry_tf not in views:
        notes.append(
            f"entry timeframe {tf_label(config.entry_tf)} unavailable; using {tf_label(entry_tf)} "
            f"(wider stops, later confirmation)"
        )
    for tf in views:
        if tf in data.derived:
            notes.append(f"{tf_label(tf)} derived by resampling {tf_label(data.base_tf)}")

    return SymbolAnalysis(
        symbol=data.symbol,
        views=views,
        bias=bias,
        bias_label=bias_label,
        bias_reason=bias_reason,
        levels=levels,
        setup=setup,
        watch=watch,
        notes=notes,
        price=price,
        session=session_label(entry.frame.index[-1]),
        atr_entry=entry.atr_now,
        entry_tf=entry_tf,
        journal=journal_snapshot(data.symbol, views, bias, bias_label, levels,
                                 config, entry_tf),
    )
