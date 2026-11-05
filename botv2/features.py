"""Market-structure and indicator features (Wilder RSI/ATR, swings, BOS/CHoCH,
liquidity sweeps, RSI divergence, ranges, session tagging)."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import time as dtime, timezone

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Indicators
# --------------------------------------------------------------------------- #

def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    """Wilder's RSI."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    avg_loss = loss.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - 100.0 / (1.0 + rs)
    # pure one-way moves: loss=0 -> 100, gain=0 -> 0 (avoids NaN -> 50)
    out = out.mask(avg_loss.eq(0) & avg_gain.gt(0), 100.0)
    out = out.mask(avg_gain.eq(0) & avg_loss.gt(0), 0.0)
    return out.fillna(50.0)


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [df["high"] - df["low"],
         (df["high"] - prev_close).abs(),
         (df["low"] - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


# --------------------------------------------------------------------------- #
# Swings & structure
# --------------------------------------------------------------------------- #

@dataclass
class Swing:
    idx: int            # position in the frame
    t: pd.Timestamp
    price: float
    kind: str           # "H" | "L"
    confirm_idx: int    # bar where the swing is knowable


@dataclass
class Event:
    idx: int
    t: pd.Timestamp
    kind: str           # "BOS" | "CHoCH"
    direction: int      # +1 bullish break, -1 bearish break
    level: float        # level that was broken
    price: float        # close at break
    label: str = ""     # e.g. "LH" / "HL" for the v2 1m BOS
    swept: float | None = None  # swept level for HL/LH sweep patterns


@dataclass
class Structure:
    trend: int                      # +1 / -1 / 0
    swings: list[Swing]             # alternating H/L used for BOS/CHoCH
    events: list[Event]
    last_swing_high: Swing | None
    last_swing_low: Swing | None
    raw_swings: list[Swing] = field(default_factory=list)   # every fractal (liquidity refs)

    def events_since(self, idx: int) -> list[Event]:
        return [e for e in self.events if e.idx >= idx]

    @property
    def last_event(self) -> Event | None:
        return self.events[-1] if self.events else None


def fractal_flags(df: pd.DataFrame, k: int = 2) -> tuple[np.ndarray, np.ndarray]:
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    n = len(df)
    sh = np.zeros(n, dtype=bool)
    sl = np.zeros(n, dtype=bool)
    for i in range(k, n - k):
        if high[i] > high[i - k:i].max() and high[i] > high[i + 1:i + k + 1].max():
            sh[i] = True
        if low[i] < low[i - k:i].min() and low[i] < low[i + 1:i + k + 1].min():
            sl[i] = True
    return sh, sl


def swing_points_raw(df: pd.DataFrame, k: int = 2) -> list[Swing]:
    sh, sl = fractal_flags(df, k)
    swings: list[Swing] = []
    for i in range(len(df)):
        if sh[i]:
            swings.append(Swing(i, df.index[i], float(df["high"].iloc[i]), "H", i + k))
        if sl[i]:
            swings.append(Swing(i, df.index[i], float(df["low"].iloc[i]), "L", i + k))
    swings.sort(key=lambda s: s.idx)
    return swings


def swing_points(df: pd.DataFrame, k: int = 2) -> list[Swing]:
    swings = swing_points_raw(df, k)
    # collapse runs of the same kind, keeping the most extreme
    cleaned: list[Swing] = []
    for s in swings:
        if cleaned and cleaned[-1].kind == s.kind:
            prev = cleaned[-1]
            better = (s.price > prev.price) if s.kind == "H" else (s.price < prev.price)
            if better:
                cleaned[-1] = s
        else:
            cleaned.append(s)
    return cleaned


def build_structure(df: pd.DataFrame, k: int = 2) -> Structure:
    """Walk the series and record BOS / CHoCH events against confirmed swings."""
    raw = swing_points_raw(df, k)
    swings = swing_points(df, k)
    by_confirm = sorted(swings, key=lambda s: s.confirm_idx)
    close = df["close"].to_numpy()
    n = len(df)

    trend = 0
    events: list[Event] = []
    last_high: Swing | None = None
    last_low: Swing | None = None
    broken_high_level: float | None = None
    broken_low_level: float | None = None
    cursor = 0

    for i in range(n):
        while cursor < len(by_confirm) and by_confirm[cursor].confirm_idx <= i:
            sw = by_confirm[cursor]
            if sw.kind == "H":
                last_high = sw
                broken_high_level = None
            else:
                last_low = sw
                broken_low_level = None
            cursor += 1

        price = close[i]
        if last_high is not None and broken_high_level != last_high.price and price > last_high.price:
            direction, level = 1, last_high.price
            kind = "CHoCH" if trend == -1 else "BOS"
            trend = 1
            broken_high_level = level
            events.append(Event(i, df.index[i], kind, direction, level, float(price)))
        if last_low is not None and broken_low_level != last_low.price and price < last_low.price:
            direction, level = -1, last_low.price
            kind = "CHoCH" if trend == 1 else "BOS"
            trend = -1
            broken_low_level = level
            events.append(Event(i, df.index[i], kind, direction, level, float(price)))

    return Structure(trend, swings, events, last_high, last_low, raw)


def choch_events(df: pd.DataFrame, structure: Structure | None = None,
                 k: int = 2) -> list[Event]:
    """v2 1m CHoCH at a trend extreme.

    Bearish context (bullish reversal): a final LL → high → higher low → close
    above that high. Bullish context (bearish reversal): a final HH → low →
    lower high → close below that low. Later highs/lows are continuation.
    """
    swings = structure.swings if structure is not None else swing_points(df, k)
    if not swings:
        return []
    by_confirm = sorted(swings, key=lambda s: s.confirm_idx)
    close = df["close"].to_numpy(dtype=float)
    events: list[Event] = []
    cursor = 0
    seq: list[Swing] = []
    last_key: dict[int, tuple[int, int, int] | None] = {1: None, -1: None}
    long_block_ll = -1
    short_block_hh = -1

    for i in range(1, len(df)):
        while cursor < len(by_confirm) and by_confirm[cursor].confirm_idx <= i:
            seq.append(by_confirm[cursor])
            cursor += 1
        if len(seq) < 5:
            continue
        a, b, c = seq[-3], seq[-2], seq[-1]
        key = (a.idx, b.idx, c.idx)
        prior = seq[-5]      # same kind as `a` because swings alternate
        prior_low = prior if prior.kind == "L" else None
        prior_high = prior if prior.kind == "H" else None

        # final LL -> high -> higher low -> break above the high (bullish)
        if (a.kind == "L" and b.kind == "H" and c.kind == "L"
                and c.price > a.price and prior_low is not None
                and a.price < prior_low.price and a.idx > long_block_ll):
            level = float(b.price)
            if last_key[1] != key and close[i] > level >= close[i - 1]:
                events.append(Event(i, df.index[i], "CHoCH", 1, level,
                                    float(close[i]), "HL"))
                last_key[1] = key
                long_block_ll = a.idx

        # final HH -> low -> lower high -> break below the low (bearish)
        if (a.kind == "H" and b.kind == "L" and c.kind == "H"
                and c.price < a.price and prior_high is not None
                and a.price > prior_high.price and a.idx > short_block_hh):
            level = float(b.price)
            if last_key[-1] != key and close[i] < level <= close[i - 1]:
                events.append(Event(i, df.index[i], "CHoCH", -1, level,
                                    float(close[i]), "LH"))
                last_key[-1] = key
                short_block_hh = a.idx

    events.sort(key=lambda event: event.idx)
    return events


def bos_reference(df: pd.DataFrame, swings: list[Swing],
                  direction: int) -> tuple[float | None, str]:
    """Most recent un-swept LH (longs) or HL (shorts) level."""
    highs = [s for s in swings if s.kind == "H"]
    lows = [s for s in swings if s.kind == "L"]
    if len(highs) < 2 or len(lows) < 2:
        return None, ""
    if direction > 0:
        # A BOS long only exists after a lower high and a lower low.
        if not (highs[-1].price < highs[-2].price
                and lows[-1].price < lows[-2].price):
            return None, ""
        same = highs
        sweep = _next_greater(df["high"].to_numpy(dtype=float))
    else:
        # A BOS short only exists after a higher high and a higher low.
        if not (highs[-1].price > highs[-2].price
                and lows[-1].price > lows[-2].price):
            return None, ""
        same = lows
        sweep = _next_lower(df["low"].to_numpy(dtype=float))
    for j in range(len(same) - 1, 0, -1):
        last, prev = same[j], same[j - 1]
        is_ref = (last.price < prev.price if direction > 0
                  else last.price > prev.price)
        if not is_ref:
            continue
        # -1 means no later candle has traded through the level.
        if sweep[last.idx] == -1:
            return float(last.price), ("LH" if direction > 0 else "HL")
    return None, ""


def _next_greater(values: np.ndarray) -> np.ndarray:
    """First index to the right with a strictly greater value (-1 if none)."""
    out = np.full(len(values), -1, dtype=int)
    stack: list[int] = []
    for i in range(len(values) - 1, -1, -1):
        while stack and values[stack[-1]] <= values[i]:
            stack.pop()
        out[i] = stack[-1] if stack else -1
        stack.append(i)
    return out


def _next_lower(values: np.ndarray) -> np.ndarray:
    """First index to the right with a strictly lower value (-1 if none)."""
    out = np.full(len(values), -1, dtype=int)
    stack: list[int] = []
    for i in range(len(values) - 1, -1, -1):
        while stack and values[stack[-1]] >= values[i]:
            stack.pop()
        out[i] = stack[-1] if stack else -1
        stack.append(i)
    return out


def rolling_atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """Simple rolling ATR used by the range detector (matches the reference)."""
    prev_close = df["close"].shift()
    tr = pd.concat([df["high"] - df["low"],
                    (df["high"] - prev_close).abs(),
                    (df["low"] - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


@dataclass
class Range:
    start_idx: int
    end_idx: int
    start_t: pd.Timestamp
    end_t: pd.Timestamp
    high: float
    low: float
    bars: int
    break_idx: int | None
    break_dir: str          # "up" | "down" | "open"


def _efficiency_ratio(close: np.ndarray) -> float:
    path = np.abs(np.diff(close)).sum()
    return float(abs(close[-1] - close[0]) / path) if path else 1.0


def detect_ranges(df: pd.DataFrame, atr_series: pd.Series,
                  min_bars: int = 24, max_bars: int = 120,
                  k_height: float = 3.0, er_max: float = 0.30,
                  tol_atr: float = 0.15, min_touch: int = 2,
                  break_atr: float = 0.25) -> list[Range]:
    """Box detector from the reference script (longest qualifying range first)."""
    if len(df) < min_bars + 2:
        return []
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)
    atr_v = atr_series.to_numpy(dtype=float)
    n = len(df)
    ranges: list[Range] = []
    i = 14 + min_bars

    while i < n:
        found = None
        for length in range(max_bars, min_bars - 1, -1):
            start = i - length + 1
            if start < 0:
                continue
            hi = float(high[start:i + 1].max())
            lo = float(low[start:i + 1].min())
            a = float(atr_v[i])
            if not np.isfinite(a) or a <= 0:
                continue
            if hi - lo > k_height * a:
                continue
            if _efficiency_ratio(close[start:i + 1]) > er_max:
                continue
            tol = tol_atr * a
            touches_top = int((high[start:i + 1] >= hi - tol).sum())
            touches_bottom = int((low[start:i + 1] <= lo + tol).sum())
            if touches_top >= min_touch and touches_bottom >= min_touch:
                found = (start, hi, lo)
                break
        if found is None:
            i += 1
            continue

        start, hi, lo = found
        j = i + 1
        while j < n and lo - break_atr * atr_v[j] <= close[j] <= hi + break_atr * atr_v[j]:
            j += 1
        end = j - 1
        box_high = float(high[start:end + 1].max())
        box_low = float(low[start:end + 1].min())
        break_dir = "open"
        if j < n:
            break_dir = "up" if close[j] > hi else "down"
        ranges.append(Range(start, end, df.index[start], df.index[end],
                            box_high, box_low, end - start + 1,
                            j if j < n else None, break_dir))
        i = j + 1
    return ranges


def resolve_range_event(df: pd.DataFrame, atr_series: pd.Series,
                        hi: float, lo: float, j0: int,
                        max_wait: int = 300, sweep_tol_atr: float = 0.05,
                        reclaim_bars: int = 3,
                        break_close_atr: float = 0.25) -> dict | None:
    """Resolve a frozen range box into a sweep / breakout / inconclusive event."""
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)
    atr_v = atr_series.to_numpy(dtype=float)
    n = len(df)
    for j in range(j0, min(n, j0 + max_wait)):
        a = float(atr_v[j]) if np.isfinite(atr_v[j]) else 0.0
        if a <= 0:
            continue
        tol = sweep_tol_atr * a
        up = high[j] > hi + tol
        down = low[j] < lo - tol
        if not (up or down):
            continue
        if up and down:
            return {"type": "ambiguous", "bar": j, "side": "both"}

        if j + reclaim_bars > n:
            return None

        side = "up" if up else "down"
        level = hi if up else lo
        ext = high[j] if up else low[j]
        for k in range(j, j + reclaim_bars):
            ext = max(ext, high[k]) if side == "up" else min(ext, low[k])
            reclaimed = close[k] <= hi if side == "up" else close[k] >= lo
            if reclaimed:
                return {"type": "sweep", "side": side, "pierce_bar": j,
                        "entry_bar": k,
                        "trade": "short" if side == "up" else "long",
                        "stop": ext + (tol if side == "up" else -tol),
                        "target": (hi + lo) / 2, "ext": ext}

        k = j + reclaim_bars - 1
        beyond = (close[k] - level) if side == "up" else (level - close[k])
        if beyond >= break_close_atr * float(atr_v[k]):
            return {"type": "breakout", "side": side, "pierce_bar": j,
                    "entry_bar": k,
                    "trade": "long" if side == "up" else "short",
                    "stop": (lo + hi) / 2}
        return {"type": "inconclusive", "side": side, "pierce_bar": j}
    return None


def range_events_from_boxes(entry_frame: pd.DataFrame,
                            source_frame: pd.DataFrame,
                            source_atr: pd.Series,
                            ranges: list[Range],
                            sweep_tol_atr: float = 0.05,
                            reclaim_bars: int = 3,
                            break_close_atr: float = 0.25,
                            max_wait: int = 300) -> tuple[list[Event], list[Sweep]]:
    """Resolve detected boxes on the source frame, then map events to the entry frame."""
    if len(entry_frame) == 0:
        return [], []
    breaks: list[Event] = []
    sweeps: list[Sweep] = []
    high = source_frame["high"].to_numpy(dtype=float)
    low = source_frame["low"].to_numpy(dtype=float)
    close = source_frame["close"].to_numpy(dtype=float)

    def entry_pos(ts) -> int | None:
        pos = int(entry_frame.index.searchsorted(ts, side="right")) - 1
        return pos if 0 <= pos < len(entry_frame) else None

    for box in ranges:
        resolved = resolve_range_event(
            source_frame, source_atr, box.high, box.low, box.end_idx + 1,
            max_wait=max_wait, sweep_tol_atr=sweep_tol_atr,
            reclaim_bars=reclaim_bars, break_close_atr=break_close_atr)
        if not resolved or resolved.get("type") not in ("sweep", "breakout"):
            continue
        entry_bar = int(resolved["entry_bar"])
        pos = entry_pos(source_frame.index[entry_bar])
        if pos is None:
            continue
        if resolved["type"] == "sweep":
            side = resolved["side"]
            direction = 1 if side == "down" else -1
            level = box.low if side == "down" else box.high
            ext = float(resolved["ext"])
            sweeps.append(Sweep(pos, entry_frame.index[pos], direction,
                                float(level), "range",
                                abs(ext - level)))
        elif resolved["type"] == "breakout":
            side = resolved["side"]
            direction = 1 if side == "up" else -1
            level = box.high if side == "up" else box.low
            breaks.append(Event(pos, entry_frame.index[pos], "Range breakout",
                                direction, float(level),
                                float(close[entry_bar]), "range"))

    breaks.sort(key=lambda event: event.idx)
    sweeps.sort(key=lambda sweep: sweep.idx)
    return breaks, sweeps


def range_breakout_events(df: pd.DataFrame, atr_series: pd.Series,
                         lookback: int = 30,
                         max_range_atr: float = 6.0) -> list[Event]:
    """1m range breakout: a close beyond a tight prior range that keeps going."""
    n = len(df)
    if n < lookback + 2:
        return []
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)
    atr = atr_series.to_numpy(dtype=float)
    prev_high = pd.Series(high).rolling(lookback).max().shift(1).to_numpy()
    prev_low = pd.Series(low).rolling(lookback).min().shift(1).to_numpy()
    events: list[Event] = []
    for i in range(1, n):
        if not (np.isfinite(atr[i]) and atr[i] > 0):
            continue
        if not (np.isfinite(prev_high[i]) and np.isfinite(prev_low[i])):
            continue
        if prev_high[i] - prev_low[i] > max_range_atr * atr[i]:
            continue
        if close[i] > prev_high[i] >= close[i - 1]:
            events.append(Event(i, df.index[i], "Range breakout", 1,
                                float(prev_high[i]), float(close[i]), "range"))
        elif close[i] < prev_low[i] <= close[i - 1]:
            events.append(Event(i, df.index[i], "Range breakout", -1,
                                float(prev_low[i]), float(close[i]), "range"))
    return events


def sweep_reversal_events(df: pd.DataFrame, structure: Structure,
                          max_wait: int = 30) -> list[Event]:
    """1m HL/LH sweep reversal.

    Long: bottom → high → higher low → high → sweep the HL → close above the
    second high. Short: top → low → lower high → low → sweep the LH → close
    below the second low.
    """
    swings = sorted(structure.swings, key=lambda s: s.confirm_idx)
    if len(swings) < 4:
        return []
    close = df["close"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    events: list[Event] = []
    cursor = 0
    seq: list[Swing] = []
    done_long: set[tuple[int, int, int, int]] = set()
    done_short: set[tuple[int, int, int, int]] = set()
    swept_long: dict[tuple[int, int, int, int], bool] = {}
    swept_short: dict[tuple[int, int, int, int], bool] = {}

    for i in range(1, len(df)):
        while cursor < len(swings) and swings[cursor].confirm_idx <= i:
            seq.append(swings[cursor])
            cursor += 1
        if len(seq) < 4:
            continue
        a, b, c, d = seq[-4], seq[-3], seq[-2], seq[-1]
        key = (a.idx, b.idx, c.idx, d.idx)

        # long: L,H,L(HL),H -> sweep the HL, then close above the second high
        if (a.kind == "L" and b.kind == "H" and c.kind == "L" and d.kind == "H"
                and c.price > a.price and key not in done_long):
            if low[i] < c.price:
                swept_long[key] = True
            if swept_long.get(key) and close[i] > d.price:
                events.append(Event(i, df.index[i], "HL sweep", 1,
                                    float(d.price), float(close[i]), "HL",
                                    float(c.price)))
                done_long.add(key)

        # short: H,L,H(LH),L -> sweep the LH, then close below the second low
        if (a.kind == "H" and b.kind == "L" and c.kind == "H" and d.kind == "L"
                and c.price < a.price and key not in done_short):
            if high[i] > c.price:
                swept_short[key] = True
            if swept_short.get(key) and close[i] < d.price:
                events.append(Event(i, df.index[i], "LH sweep", -1,
                                    float(d.price), float(close[i]), "LH",
                                    float(c.price)))
                done_short.add(key)

    return events


def unswept_indices(frame: pd.DataFrame, swings: list[Swing]) -> set[int]:
    """Swing indices whose level no later candle has traded through."""
    high = frame["high"].to_numpy(dtype=float)
    low = frame["low"].to_numpy(dtype=float)
    n = len(frame)
    suffix_max = np.maximum.accumulate(high[::-1])[::-1]
    suffix_min = np.minimum.accumulate(low[::-1])[::-1]
    out: set[int] = set()
    for swing in swings:
        if swing.idx + 1 >= n:
            out.add(swing.idx)
        elif swing.kind == "H":
            if suffix_max[swing.idx + 1] < swing.price:
                out.add(swing.idx)
        elif suffix_min[swing.idx + 1] > swing.price:
            out.add(swing.idx)
    return out


def bos_events(df: pd.DataFrame, structure: Structure | None = None,
               k: int = 2) -> list[Event]:
    """v2 1m BOS at a trend extreme.

    Long: after a final lower low and a lower high, close above the most recent
    un-swept LH. Short: after a final higher high and a higher low, close below
    the most recent un-swept HL. A level touched by an earlier candle is stale.
    Further highs/lows after the BOS are continuation, not new BOS events.
    """
    swings = structure.swings if structure is not None else swing_points(df, k)
    if not swings:
        return []
    by_confirm = sorted(swings, key=lambda s: s.confirm_idx)
    close = df["close"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    next_high = _next_greater(high)
    next_low = _next_lower(low)
    events: list[Event] = []
    cursor = 0
    prev_high = last_high = prev_low = last_low = None
    long_block_ll = -1
    short_block_hh = -1
    lh_stack: list[tuple[float, int]] = []     # (level, first sweep index)
    hl_stack: list[tuple[float, int]] = []

    for i in range(1, len(df)):
        while cursor < len(by_confirm) and by_confirm[cursor].confirm_idx <= i:
            swing = by_confirm[cursor]
            if swing.kind == "H":
                if last_high is not None and swing.price < last_high.price:
                    lh_stack.append((float(swing.price), int(next_high[swing.idx])))
                prev_high, last_high = last_high, swing
            else:
                if last_low is not None and swing.price > last_low.price:
                    hl_stack.append((float(swing.price), int(next_low[swing.idx])))
                prev_low, last_low = last_low, swing
            cursor += 1

        # A level is still usable on the bar that first sweeps it; it becomes
        # stale from the following bar onwards.
        while lh_stack and lh_stack[-1][1] != -1 and lh_stack[-1][1] < i:
            lh_stack.pop()
        while hl_stack and hl_stack[-1][1] != -1 and hl_stack[-1][1] < i:
            hl_stack.pop()

        # Only a BOS at the trend extreme: a final LL/LH (or HH/HL) sequence.
        long_extreme = (last_high is not None and prev_high is not None
                        and last_high.price < prev_high.price
                        and last_low is not None and prev_low is not None
                        and last_low.price < prev_low.price)
        short_extreme = (last_low is not None and prev_low is not None
                         and last_low.price > prev_low.price
                         and last_high is not None and prev_high is not None
                         and last_high.price > prev_high.price)

        if long_extreme and last_low.idx > long_block_ll and lh_stack:
            level = lh_stack[-1][0]
            if close[i] > level >= close[i - 1]:
                events.append(Event(i, df.index[i], "BOS", 1, level,
                                    float(close[i]), "LH"))
                lh_stack.pop()
                long_block_ll = last_low.idx

        if short_extreme and last_high.idx > short_block_hh and hl_stack:
            level = hl_stack[-1][0]
            if close[i] < level <= close[i - 1]:
                events.append(Event(i, df.index[i], "BOS", -1, level,
                                    float(close[i]), "HL"))
                hl_stack.pop()
                short_block_hh = last_high.idx

    events.sort(key=lambda event: event.idx)
    return events


# --------------------------------------------------------------------------- #
# Liquidity sweeps & ranges
# --------------------------------------------------------------------------- #

@dataclass
class Sweep:
    idx: int
    t: pd.Timestamp
    direction: int          # +1 bullish (low swept), -1 bearish (high swept)
    level: float
    source: str             # "swing" | "range"
    wick: float             # how far beyond the level


def detect_sweeps(df: pd.DataFrame, atr_series: pd.Series, structure: Structure,
                  lookback: int = 20,
                  swings: list[Swing] | None = None) -> list[Sweep]:
    """Liquidity grabs: wick through a swing/range level and close back inside.

    Vectorised — this runs on every 1-minute update, so it must stay cheap even on
    65k-bar export files.
    """
    n = len(df)
    if n < 5:
        return []
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)
    atr = atr_series.to_numpy(dtype=float)

    swings = swings or structure.raw_swings or structure.swings
    active_low = np.full(n, np.nan)
    active_high = np.full(n, np.nan)
    # a swing level is "live" for the next `lookback` bars after it prints
    for swing in swings:
        lo = swing.idx + 1
        hi = min(swing.idx + lookback + 1, n)
        if lo >= hi:
            continue
        if swing.kind == "L":
            active_low[lo:hi] = swing.price        # newer lows overwrite older ones
        else:
            active_high[lo:hi] = swing.price

    prev_high = pd.Series(high).rolling(30).max().shift(1).to_numpy()
    prev_low = pd.Series(low).rolling(30).min().shift(1).to_numpy()
    tight_range = (prev_high - prev_low) <= 6.0 * atr

    bull_swing = (low < active_low) & (close > active_low)
    bear_swing = (high > active_high) & (close < active_high)
    bull_range = tight_range & (low < prev_low) & (close > prev_low)
    bear_range = tight_range & (high > prev_high) & (close < prev_high)

    out: list[Sweep] = []
    for i in range(2, n):
        if not np.isfinite(atr[i]) or atr[i] <= 0:
            continue
        ts = df.index[i]
        if bull_swing[i]:
            out.append(Sweep(i, ts, 1, float(active_low[i]), "swing",
                             float(active_low[i] - low[i])))
        elif bull_range[i]:
            out.append(Sweep(i, ts, 1, float(prev_low[i]), "range",
                             float(prev_low[i] - low[i])))
        elif bear_swing[i]:
            out.append(Sweep(i, ts, -1, float(active_high[i]), "swing",
                             float(high[i] - active_high[i])))
        elif bear_range[i]:
            out.append(Sweep(i, ts, -1, float(prev_high[i]), "range",
                             float(high[i] - prev_high[i])))
    return out


# --------------------------------------------------------------------------- #
# RSI divergence
# --------------------------------------------------------------------------- #

@dataclass
class Divergence:
    idx: int            # bar where it becomes confirmable
    t: pd.Timestamp
    direction: int      # +1 bullish, -1 bearish
    p1: float
    p2: float
    rsi1: float
    rsi2: float
    equal_price: bool

    @property
    def label(self) -> str:
        if self.equal_price:
            return "price EQ / RSI HL" if self.direction > 0 else "price EQ / RSI LH"
        return "price LL / RSI HL" if self.direction > 0 else "price HH / RSI LH"


def detect_divergences(df: pd.DataFrame, rsi_series: pd.Series, structure: Structure,
                       min_bars: int = 5, equal_tol_atr: float = 0.35,
                       rsi_tol: float = 0.8,
                       swings: list[Swing] | None = None) -> list[Divergence]:
    atr_series = atr(df, 14)
    out: list[Divergence] = []

    def pair(swings: list[Swing], direction: int) -> None:
        for a, b in zip(swings, swings[1:]):
            if b.idx - a.idx < min_bars:
                continue
            r1, r2 = float(rsi_series.iloc[a.idx]), float(rsi_series.iloc[b.idx])
            tol = equal_tol_atr * float(atr_series.iloc[b.idx] or 0.0)
            if direction > 0:
                lower = b.price < a.price - 1e-12
                equal = tol > 0 and abs(b.price - a.price) <= tol
                if (lower or equal) and r2 > r1 + rsi_tol:
                    out.append(Divergence(b.confirm_idx, df.index[min(b.confirm_idx, len(df) - 1)],
                                          1, a.price, b.price, r1, r2, equal))
            else:
                higher = b.price > a.price + 1e-12
                equal = tol > 0 and abs(b.price - a.price) <= tol
                if (higher or equal) and r2 < r1 - rsi_tol:
                    out.append(Divergence(b.confirm_idx, df.index[min(b.confirm_idx, len(df) - 1)],
                                          -1, a.price, b.price, r1, r2, equal))

    source = swings or structure.swings
    pair([s for s in source if s.kind == "L"], 1)
    pair([s for s in source if s.kind == "H"], -1)
    out.sort(key=lambda d: d.idx)
    return out


# --------------------------------------------------------------------------- #
# Sessions
# --------------------------------------------------------------------------- #

def session_label(ts_utc: pd.Timestamp) -> str:
    hour = ts_utc.hour
    if 0 <= hour < 7:
        return "Asia"
    if 7 <= hour < 12:
        return "London"
    if 12 <= hour < 13:
        return "London/NY overlap"
    if 13 <= hour < 17:
        return "New York"
    if 17 <= hour < 21:
        return "NY late"
    return "After-hours"


def ny_open_window(ts_utc: pd.Timestamp) -> bool:
    """13:00-15:30 UTC approximates the NY open impulse the journal trades."""
    t = ts_utc.time()
    return dtime(13, 0) <= t <= dtime(15, 30)


def now_utc() -> pd.Timestamp:
    return pd.Timestamp.now(tz=timezone.utc)

# --------------------------------------------------------------------------- #
# Head & shoulders (1-min patterns the journal actually trades)
# --------------------------------------------------------------------------- #

@dataclass
class HSPattern:
    kind: str                # "regular" (topping) | "inverse" (bottoming)
    direction: int           # -1 for regular, +1 for inverse
    left_idx: int
    left_price: float
    head_idx: int
    head_price: float
    right_idx: int
    right_price: float
    neckline: float
    break_idx: int | None = None
    bars_since_break: int | None = None

    @property
    def complete(self) -> bool:
        return self.break_idx is not None

    @property
    def label(self) -> str:
        base = "inverse H&S" if self.direction > 0 else "H&S"
        return f"{base} (neck {self.neckline:.5f})"


def detect_head_shoulders(df: pd.DataFrame, structure: Structure, atr_series: pd.Series,
                          lookback: int = 120, shoulder_tol_atr: float = 0.9,
                          head_margin_atr: float = 0.15,
                          break_window: int = 15) -> list[HSPattern]:
    """Detect the most recent (inverse) head & shoulders patterns.

    A pattern is complete only once a candle *closes* through the neckline after the
    right shoulder is confirmed — the journal's lesson about entering before the right
    shoulder formed is encoded as ``complete is False``.
    """
    swings = structure.swings
    if len(swings) < 5 or len(df) < 30:
        return []

    close = df["close"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    index = df.index.to_numpy()
    last = len(df) - 1
    patterns: list[HSPattern] = []

    def atr_at(index: int) -> float:
        value = float(atr_series.iloc[index]) if index < len(atr_series) else 0.0
        return value if np.isfinite(value) and value > 0 else 0.0

    def neckline_between(left: Swing, head: Swing, right: Swing, direction: int) -> float:
        if direction > 0:      # inverse: neckline is the highest high between the lows
            first = float(high[left.idx:head.idx + 1].max())
            second = float(high[head.idx:right.idx + 1].max())
            return min(first, second)
        first = float(low[left.idx:head.idx + 1].min())
        second = float(low[head.idx:right.idx + 1].min())
        return max(first, second)

    for wanted_kind, direction in (("H", -1), ("L", 1)):
        seq = [sw for sw in swings if sw.kind == wanted_kind]
        for left, head, right in zip(seq, seq[1:], seq[2:]):
            if right.idx - left.idx > lookback or (last - right.idx) > lookback:
                continue
            atr_ref = atr_at(right.idx)
            if atr_ref <= 0:
                continue
            shoulders_close = abs(left.price - right.price) <= shoulder_tol_atr * atr_ref
            if not shoulders_close:
                continue
            if direction < 0:
                is_head = head.price > max(left.price, right.price) + head_margin_atr * atr_ref
            else:
                is_head = head.price < min(left.price, right.price) - head_margin_atr * atr_ref
            if not is_head:
                continue

            neckline = neckline_between(left, head, right, direction)
            if neckline <= 0:
                continue

            break_idx = None
            start = max(right.confirm_idx, right.idx + 1)
            for i in range(start, min(start + break_window + 1, len(df))):
                price = float(close[i])
                if direction < 0 and price < neckline:
                    break_idx = i
                    break
                if direction > 0 and price > neckline:
                    break_idx = i
                    break
            if break_idx is not None and (last - break_idx) > break_window:
                continue

            if break_idx is not None:
                patterns.append(HSPattern(
                    kind="regular" if direction < 0 else "inverse", direction=direction,
                    left_idx=left.idx, left_price=left.price,
                    head_idx=head.idx, head_price=head.price,
                    right_idx=right.idx, right_price=right.price,
                    neckline=neckline, break_idx=break_idx,
                    bars_since_break=last - break_idx,
                ))

    patterns.sort(key=lambda p: p.right_idx)
    return patterns[-3:] if patterns else []


# --------------------------------------------------------------------------- #
# Candle patterns (the "interesting candles" part of the 1-min read)
# --------------------------------------------------------------------------- #

@dataclass
class CandlePattern:
    idx: int
    t: pd.Timestamp
    kind: str          # hammer | shooting_star | bull_engulf | bear_engulf | capitulation
    direction: int     # +1 bullish, -1 bearish


def detect_candle_patterns(df: pd.DataFrame, atr_series: pd.Series) -> list[CandlePattern]:
    """Wick/body anomalies worth marking on the entry chart."""
    o = df["open"].to_numpy(float)
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    atr = atr_series.to_numpy(float)
    out: list[CandlePattern] = []
    for i in range(1, len(df)):
        body = abs(c[i] - o[i])
        rng = h[i] - l[i]
        if rng <= 0 or not (atr[i] > 0):
            continue
        upper = h[i] - max(o[i], c[i])
        lower = min(o[i], c[i]) - l[i]
        close_pos = (c[i] - l[i]) / rng
        kind, direction = None, 0

        prev_body = abs(c[i - 1] - o[i - 1])
        if body >= 2.0 * atr[i] and body / rng >= 0.8 and rng >= 2.0 * atr[i]:
            kind, direction = "capitulation", (1 if c[i] > o[i] else -1)
        elif (c[i] > o[i] and c[i - 1] < o[i - 1] and body >= 0.6 * atr[i]
              and c[i] >= o[i - 1] and o[i] <= c[i - 1] and body >= 1.2 * prev_body):
            kind, direction = "bull_engulf", 1
        elif (c[i] < o[i] and c[i - 1] > o[i - 1] and body >= 0.6 * atr[i]
              and c[i] <= o[i - 1] and o[i] >= c[i - 1] and body >= 1.2 * prev_body):
            kind, direction = "bear_engulf", -1
        elif (lower >= 2.5 * body and upper <= 0.5 * body and close_pos >= 0.65
              and rng >= 0.8 * atr[i]):
            kind, direction = "hammer", 1
        elif (upper >= 2.5 * body and lower <= 0.5 * body and close_pos <= 0.35
              and rng >= 0.8 * atr[i]):
            kind, direction = "shooting_star", -1
        if kind:
            out.append(CandlePattern(i, df.index[i], kind, direction))
    return out


def detect_reversal_continuation(df: pd.DataFrame, atr_series: pd.Series,
                                 trend: int) -> list[int]:
    """1-min bullish reversal candles *inside* an up-move (the journal's entry trigger).

    Bullish candle, long lower wick, closing strong, after a short pullback, with the
    5-min structure still bullish.
    """
    if trend <= 0:
        return []
    o = df["open"].to_numpy(float)
    c = df["close"].to_numpy(float)
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    idxs: list[int] = []
    for i in range(3, len(df)):
        body = abs(c[i] - o[i])
        rng = h[i] - l[i]
        if rng <= 0 or c[i] <= o[i]:
            continue
        lower = min(o[i], c[i]) - l[i]
        if lower < 1.5 * body:
            continue
        if (c[i] - l[i]) / rng < 0.55:
            continue
        if not (c[i - 1] < c[i - 2] and c[i - 2] < c[i - 3]):
            continue
        idxs.append(i)
    return idxs


# --------------------------------------------------------------------------- #
# Real swings (pivot + prominence + isolation)
# --------------------------------------------------------------------------- #

def real_swings(df: pd.DataFrame, atr_series: pd.Series,
                n: int = 5, prom: float = 1.0, iso_band: float = 0.35,
                prom_window: int | None = None,
                iso_window: int | None = None) -> list[Swing]:
    """Swing highs/lows that actually stand out from the surrounding price.

    A pivot must pass three filters (swing lows are the mirror image):

    1. pivot      — strictly higher than the ``n`` bars on each side (ties fail);
    2. prominence — price falls away by at least ``prom`` ATR on *both* sides,
                    measured against the lowest low within ``prom_window`` bars
                    each side (default ``2 * n``);
    3. isolation  — no other candle high comes within ``iso_band`` ATR of the
                    pivot, looking ``iso_window`` bars each side (default
                    ``5 * n``); the ``n`` core bars are skipped because the pivot
                    filter already makes them lower.

    ``n=4`` for 1H and ``n=5`` for 5M reproduce the reference detector.
    """
    n = max(1, int(n))
    prom = float(prom)
    iso_band = float(iso_band)
    pw = max(1, int(prom_window if prom_window is not None else 2 * n))
    iw = max(1, int(iso_window if iso_window is not None else 5 * n))

    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    atr_v = atr_series.to_numpy(dtype=float)
    total = len(df)
    if total <= 2 * n:
        return []

    swings: list[Swing] = []
    for kind, ext, opp, sign in (("H", high, low, 1), ("L", -low, -high, -1)):
        for i in range(n, total - n):
            a = float(atr_v[i])
            if not np.isfinite(a) or a <= 0:
                continue
            pivot = ext[i]
            # 1. strict pivot: every core bar on both sides must be lower
            if not ((ext[i - n:i] < pivot).all()
                    and (ext[i + 1:i + n + 1] < pivot).all()):
                continue
            # 2. prominence on both sides, against the opposite extreme
            left = (pivot - opp[max(0, i - pw):i].min()) / a
            right = (pivot - opp[i + 1:i + pw + 1].min()) / a
            if left < prom or right < prom:
                continue
            # 3. isolation: no comparable extreme nearby (skip the core bars)
            others = np.concatenate([
                ext[max(0, i - iw):max(0, i - n)],
                ext[i + n + 1:min(total, i + iw + 1)],
            ])
            if len(others) and np.abs(others - pivot).min() / a <= iso_band:
                continue
            swings.append(Swing(i, df.index[i], float(sign * pivot), kind, i + n))

    swings.sort(key=lambda s: s.idx)
    return swings


# --------------------------------------------------------------------------- #
# Major swings (legacy filter for the other timeframes)
# --------------------------------------------------------------------------- #

def major_swings(df: pd.DataFrame, atr_series: pd.Series,
                 swings: list[Swing] | None = None,
                 atr_mult: float = 1.2, min_gap: int = 6,
                 max_touches: int = 3, tol_atr: float = 0.35,
                 window: int = 80) -> list[Swing]:
    """Keep only swings that stand out in price and are not part of a wick cluster.

    Three filters:
      1. prominence — the peak/trough must clear the adjacent valley/peak by
         ``atr_mult`` ATRs (no micro-wiggles);
      2. cleanliness — at most ``max_touches`` other candles may have wicks within
         ``tol_atr`` ATR of the level in the surrounding ``window`` bars (a level
         peppered with wicks is chop, not a swing);
      3. spacing — two accepted swings of the same kind must be ``min_gap`` bars apart.
    """
    if swings is None:
        swings = swing_points_raw(df, 2)
    n = len(df)
    if n < 5 or not swings:
        return []
    high = df["high"].to_numpy(float)
    low = df["low"].to_numpy(float)
    atr = atr_series.to_numpy(float)
    look = max(2, min_gap)

    accepted: list[Swing] = []
    for swing in sorted(swings, key=lambda s: s.idx):
        i = swing.idx
        if i < 1 or i >= n - 1:
            continue
        a = float(atr[i]) if i < len(atr) else 0.0
        if not np.isfinite(a) or a <= 0:
            continue

        left_lo = float(low[max(0, i - look):i].min()) if i > 0 else swing.price
        right_lo = float(low[i + 1:min(n, i + 1 + look)].min()) if i + 1 < n else swing.price
        left_hi = float(high[max(0, i - look):i].max()) if i > 0 else swing.price
        right_hi = float(high[i + 1:min(n, i + 1 + look)].max()) if i + 1 < n else swing.price
        if not (np.isfinite(left_lo) and np.isfinite(right_lo)):
            left_lo, right_lo = swing.price, swing.price

        tol = tol_atr * a
        lo_idx = max(0, i - window)
        hi_idx = min(n, i + window)
        positions = np.arange(lo_idx, hi_idx)
        far = np.abs(positions - i) > 3

        if swing.kind == "H":
            valley = max(left_lo, right_lo)
            if swing.price - valley < atr_mult * a:
                continue
            touches = int((high[positions[far]] >= swing.price - tol).sum())
        else:
            peak = min(left_hi, right_hi)
            if peak - swing.price < atr_mult * a:
                continue
            touches = int((low[positions[far]] <= swing.price + tol).sum())
        if touches > max_touches:
            continue

        # spacing: replace the previous same-kind swing if this one is more extreme
        same_kind = [s for s in accepted if s.kind == swing.kind]
        if same_kind and i - same_kind[-1].idx < min_gap:
            prev = same_kind[-1]
            better = (swing.price > prev.price) if swing.kind == "H" else (swing.price < prev.price)
            if better:
                accepted.remove(prev)
                accepted.append(swing)
            continue
        accepted.append(swing)

    accepted.sort(key=lambda s: s.idx)
    return accepted
