"""Data layer: MT4 CSV ingestion, timeframe discovery and resampling."""
from __future__ import annotations

import math
import re
import zlib
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config

TF_NAMES = {1: "1m", 5: "5m", 15: "15m", 30: "30m", 60: "1h", 240: "4h", 1440: "1d"}


def tf_label(minutes: int) -> str:
    return TF_NAMES.get(minutes, f"{minutes}m")


@dataclass
class SymbolData:
    symbol: str
    frames: dict[int, pd.DataFrame]           # tf minutes -> OHLCV (UTC index)
    base_tf: int                              # granularity of the raw file
    derived: set[int] = field(default_factory=set)
    files: dict[int, Path] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    server_offset: float | None = None

    @property
    def available(self) -> tuple[int, ...]:
        return tuple(sorted(self.frames))

    def df(self, tf: int) -> pd.DataFrame | None:
        return self.frames.get(tf)

    def last_bar_age_minutes(self, now: pd.Timestamp | None = None) -> float:
        if not self.frames:
            return float("inf")
        now = now or pd.Timestamp.now(tz="UTC")
        last = max(df.index[-1] for df in self.frames.values())
        return float((now - last).total_seconds() / 60.0)

    def coverage(self) -> str:
        if not self.frames:
            return "no data"
        first = min(df.index[0] for df in self.frames.values())
        last = max(df.index[-1] for df in self.frames.values())
        bars = len(self.frames[min(self.frames)])
        return f"{bars} bars @{tf_label(self.base_tf)}, {first:%Y-%m-%d %H:%M} → {last:%Y-%m-%d %H:%M} UTC"


# --------------------------------------------------------------------------- #
# MT4 CSV parsing
# --------------------------------------------------------------------------- #

_NUM_RE = re.compile(r"^[+-]?\d+(?:[.,]\d+)?$")
_TIME_FULL = re.compile(r"^\d{1,2}[:.]\d{2}(?::\d{2})?$")
_DATE_FULL = re.compile(r"^\d{1,4}[./-]\d{1,2}[./-]\d{1,4}$")


def _detect_sep(line: str) -> str:
    counts = {sep: line.count(sep) for sep in (";", ",", "\t")}
    return max(counts, key=counts.get)


def _num(raw: str, decimal_comma: bool = False) -> float:
    raw = raw.strip().replace("\ufeff", "")
    if decimal_comma and raw.count(",") == 1 and "." not in raw:
        raw = raw.replace(",", ".")
    return float(raw)


def _looks_like_header(fields: list[str]) -> bool:
    joined = " ".join(field.strip().lower() for field in fields)
    markers = ("date", "time", "open", "high", "low", "close", "volume", "tickvol", "<")
    return any(marker in joined for marker in markers)


def _parse_datetime(date_raw: str, time_raw: str | None) -> datetime | None:
    date_raw = date_raw.strip().replace("\ufeff", "")
    time_raw = (time_raw or "").strip()
    parts = re.split(r"[./\-]", date_raw)
    if len(parts) != 3:
        return None
    try:
        a, b, c = (int(p) for p in parts)
    except ValueError:
        return None
    if a > 99:                      # 2026.10.25
        year, month, day = a, b, c
    elif c > 99:                    # 25.10.2026
        day, month, year = a, b, c
    else:                           # 26.10.25
        year = 2000 + a if a < 70 else 1900 + a
        day, month = b, c
    hour = minute = second = 0
    if time_raw:
        tparts = re.split(r"[:.]", time_raw)
        try:
            nums = [int(p) for p in tparts if p != ""]
        except ValueError:
            return None
        if len(nums) >= 2:
            hour, minute = nums[0], nums[1]
            second = nums[2] if len(nums) >= 3 else 0
        elif len(time_raw) in (3, 4):   # 1430 / 930
            hour, minute = int(time_raw[:-2]), int(time_raw[-2:])
    try:
        return datetime(year, month, day, hour, minute, min(second, 59))
    except ValueError:
        return None


def infer_server_offset(index: pd.DatetimeIndex, now: datetime | None = None) -> int | None:
    """Guess the broker's UTC offset from how close the last bar is to 'now'.

    Only trusts the inference when the export is recent (within 36h), otherwise the
    configured offset is used.
    """
    if len(index) == 0:
        return None
    now = now or datetime.now(timezone.utc).replace(tzinfo=None)
    delta_hours = (index[-1].to_pydatetime() - now).total_seconds() / 3600.0
    if abs(delta_hours) > 36:
        return None
    offset = int(round(delta_hours))
    return max(-2, min(4, offset))


def parse_mt4_csv(path: Path, server_utc_offset_hours: float = 2.0,
                  auto_offset: bool = False) -> tuple[pd.DataFrame, int]:
    """Parse an MT4/MQL4 history export into a UTC-indexed OHLCV frame.

    Handles comma/semicolon/tab separators, decimal commas, header or no header,
    ``Date,Time,Open,High,Low,Close,Volume[,Spread]`` and single datetime columns.
    """
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"{path.name}: file is empty")
    sep = _detect_sep(lines[0])
    rows: list[tuple[datetime, float, float, float, float, float]] = []

    start = 0
    first_fields = [f.strip() for f in lines[0].split(sep)]
    if _looks_like_header(first_fields):
        start = 1

    decimal_comma = sep == ";"

    for line in lines[start:]:
        fields = [f.strip() for f in line.split(sep)]
        dt: datetime | None = None
        nums: list[float] = []

        # Single datetime column: "2026.10.25 14:30"
        chunk = fields[0]
        pieces = chunk.split()
        if len(pieces) >= 2:
            dt = _parse_datetime(pieces[0], pieces[1])
            rest = fields[1:]
        # Separate date + time columns: "2026.10.25,14:30"
        elif len(fields) >= 6 and _TIME_FULL.match(fields[1]):
            dt = _parse_datetime(fields[0], fields[1])
            rest = fields[2:]
        # Date only (daily exports)
        elif _DATE_FULL.match(chunk):
            dt = _parse_datetime(chunk, None)
            rest = fields[1:]
        else:
            continue

        if dt is None:
            continue
        for raw in rest:
            try:
                nums.append(_num(raw, decimal_comma))
            except ValueError:
                break
        if len(nums) < 5:
            continue

        open_, high, low, close = nums[0], nums[1], nums[2], nums[3]
        volume = nums[4] if len(nums) > 4 else 0.0
        if not all(math.isfinite(v) for v in (open_, high, low, close)):
            continue
        if high < low:
            continue
        rows.append((dt, open_, high, low, close, volume))

    if not rows:
        raise ValueError(f"{path.name}: no parseable rows (separator={sep!r})")

    index = pd.DatetimeIndex([r[0] for r in rows])
    values = np.array([r[1:] for r in rows], dtype=float)
    df = pd.DataFrame(values, index=index, columns=["open", "high", "low", "close", "volume"])
    df = df[~df.index.duplicated(keep="last")].sort_index()

    # MT4 server time -> UTC
    offset = server_utc_offset_hours
    if auto_offset:
        inferred = infer_server_offset(df.index)
        if inferred is not None:
            offset = inferred
    df.index = (df.index - pd.Timedelta(hours=offset)).tz_localize("UTC")

    # sanity: drop bars with non-positive prices
    df = df[(df[["open", "high", "low", "close"]] > 0).all(axis=1)]
    if len(df) == 0:
        raise ValueError(f"{path.name}: no valid bars after cleaning")
    return df, offset


# --------------------------------------------------------------------------- #
# File discovery
# --------------------------------------------------------------------------- #

_STEM_RE = re.compile(r"^(?P<sym>.+?)[ _\-.]?(?:(?P<prefix>[MHDW]))?(?P<num>\d+)$", re.IGNORECASE)
_STEM_RE_SUFFIX = re.compile(r"^(?P<sym>.+?)[ _\-.]?(?P<num>\d+)(?P<suffix>[MHDW])$", re.IGNORECASE)


def infer_timeframe(path: Path) -> tuple[str, int] | None:
    """Return (symbol, timeframe-minutes) from a filename like ``XAUUSD_5.csv``."""
    stem = path.stem.strip()
    match = _STEM_RE_SUFFIX.match(stem)
    if match:
        symbol = match.group("sym").upper().replace("_", "")
        num = int(match.group("num"))
        prefix = match.group("suffix").upper()
    else:
        match = _STEM_RE.match(stem)
        if not match:
            return None
        symbol = match.group("sym").upper().replace("_", "")
        num = int(match.group("num"))
        prefix = (match.group("prefix") or "").upper()
    if prefix == "M":
        minutes = num
    elif prefix == "H":
        minutes = num * 60
    elif prefix == "D":
        minutes = num * 1440
    elif prefix == "W":
        minutes = num * 10080
    else:
        minutes = num
    if minutes not in (1, 5, 15, 30, 60, 240, 1440):
        return None
    return symbol, minutes


def discover(data_dir: Path, symbols: tuple[str, ...] | None = None) -> dict[str, dict[int, Path]]:
    """Scan ``data_dir`` for MT4 CSVs and group them by symbol/timeframe."""
    found: dict[str, dict[int, Path]] = {}
    if not data_dir.exists():
        return found
    for path in sorted(data_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() != ".csv":
            continue
        parsed = infer_timeframe(path)
        if not parsed:
            continue
        symbol, minutes = parsed
        if symbols and symbol not in symbols:
            continue
        found.setdefault(symbol, {})[minutes] = path
    return found


def resample(df: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """Aggregate OHLCV to a coarser timeframe (left-closed, epoch aligned)."""
    if minutes <= 1:
        return df
    rule = f"{minutes}min"
    out = df.resample(rule, origin="epoch", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    )
    return out.dropna(subset=["open", "high", "low", "close"])


def load_symbol(
    symbol: str,
    files: dict[int, Path],
    config: Config,
    wanted_tfs: tuple[int, ...] | None = None,
) -> SymbolData | None:
    """Load the finest file for a symbol and derive the coarser frames."""
    if not files:
        return None
    wanted = wanted_tfs or (240, 60, 15, 5)
    base_tf = min(files)
    base_path = files[base_tf]
    base, offset = parse_mt4_csv(base_path, config.server_utc_offset_hours,
                                 config.auto_server_offset)
    notes: list[str] = []
    notes.append(f"server clock converted to UTC with offset +{offset}h")

    frames: dict[int, pd.DataFrame] = {base_tf: base}
    derived: set[int] = set()
    used_files: dict[int, Path] = {base_tf: base_path}

    for tf in wanted:
        if tf == base_tf:
            continue
        if tf in files and tf != base_tf:      # a native file exists and is coarser
            try:
                frames[tf], _ = parse_mt4_csv(files[tf], config.server_utc_offset_hours,
                                              config.auto_server_offset)
                used_files[tf] = files[tf]
                continue
            except Exception:
                pass
        if tf > base_tf:
            frames[tf] = resample(base, tf)
            derived.add(tf)
            used_files[tf] = base_path
        else:
            notes.append(
                f"no {tf_label(tf)} data (finest available is {tf_label(base_tf)}); "
                f"{tf_label(tf)}-precision triggers are unavailable"
            )

    for tf, frame in frames.items():
        if len(frame) < config.min_bars:
            notes.append(f"{tf_label(tf)} has only {len(frame)} bars")

    return SymbolData(
        symbol=symbol,
        frames=frames,
        base_tf=base_tf,
        derived=derived,
        files=used_files,
        notes=notes,
        server_offset=offset,
    )


def load_all(config: Config) -> dict[str, SymbolData]:
    discovered = discover(config.data_dir, tuple(config.symbols))
    out: dict[str, SymbolData] = {}
    for symbol in config.symbols:
        files = discovered.get(symbol, {})
        if not files:
            continue
        wanted = tuple(sorted({*config.context_tfs, *config.setup_tfs, config.entry_tf}))
        data = load_symbol(symbol, files, config, wanted)
        if data:
            out[symbol] = data
    return out


# --------------------------------------------------------------------------- #
# Demo fixtures (synthetic, MT4 format) - for tests and first-run demo
# --------------------------------------------------------------------------- #

_SCALES = {
    "XAUUSD": (2650.0, 0.9),
    "EURUSD": (1.0850, 0.00020),
    "BTCUSD": (67000.0, 22.0),
    "GBPUSD": (1.2950, 0.00022),
    "USDJPY": (150.50, 0.030),
}


def _scenario(n: int, drift: float, noise: float, rng: np.random.Generator) -> list[float]:
    closes = [0.0]
    for i in range(n):
        closes.append(closes[-1] + drift + rng.normal(0, noise) + 0.12 * drift * math.sin(i / 70.0))
    return closes[1:]


DEFAULT_SCENARIO = {
    "pull": (6, 0.95),      # bars, step multiplier
    "bounce": (2, 0.65),
    "grind": (6, 0.25),
    "sweep_reclaim": 1.00,
    "reclaim": (3, 0.75),
    "choch_off": 0.20,
    "holder": 0,
}


def synth_closes(symbol: str, base_tf: int, bars: int, direction: int = 1,
                 end: datetime | None = None,
                 scenario: dict | None = None) -> tuple[list[datetime], np.ndarray]:
    """Build a deterministic series that ends in a textbook sweep + CHoCH scenario."""
    base, step = _SCALES.get(symbol.upper(), (100.0, 0.05))
    rng = np.random.default_rng(zlib.crc32(symbol.upper().encode()))

    warm = max(400, bars - 22)
    closes = _scenario(warm, drift=direction * 0.055 * step, noise=0.32 * step, rng=rng)
    flat = [c + base for c in closes]

    # ---- scenario: pullback -> sweep -> fresh CHoCH in the trend direction ---- #
    p = {**DEFAULT_SCENARIO, **(scenario or {})}
    d = direction
    c0 = flat[-1]
    pull_bars, pull_step = p["pull"]
    bounce_bars, bounce_step = p["bounce"]
    grind_bars, grind_step = p["grind"]
    reclaim_bars, reclaim_step = p["reclaim"]
    choch_off = p["choch_off"]
    holder = p["holder"]

    pull = [c0 - d * pull_step * step * k for k in range(1, pull_bars + 1)]
    l1 = pull[-1]
    bounce = [l1 + d * bounce_step * step * k for k in range(1, bounce_bars + 1)]
    neckline = bounce[-1]
    grind = [bounce[-1] - d * grind_step * step * k for k in range(1, grind_bars + 1)]
    l2 = grind[-1]
    sweep_close = l2 + d * p["sweep_reclaim"] * step
    reclaim = [sweep_close + d * reclaim_step * step * k
               for k in range(1, reclaim_bars + 1)]
    choch_close = neckline + d * choch_off * step
    tail = [choch_close + d * 0.30 * step * k for k in range(1, holder + 1)]

    flat.extend(pull + bounce + grind + [sweep_close] + reclaim + [choch_close] + tail)

    bars = len(flat)
    end = end or datetime.utcnow().replace(second=0, microsecond=0)
    end = end.replace(minute=end.minute - end.minute % base_tf)
    index = [end - timedelta(minutes=base_tf * (bars - 1 - i)) for i in range(bars)]
    return index, np.array(flat, dtype=float)


def synth_frame(symbol: str, base_tf: int = 5, bars: int = 900,
                direction: int = 1, end: datetime | None = None,
                scenario: dict | None = None) -> pd.DataFrame:
    index, close = synth_closes(symbol, base_tf, bars, direction, end, scenario)
    base, step = _SCALES.get(symbol.upper(), (100.0, 0.05))
    rng = np.random.default_rng(zlib.crc32((symbol.upper() + "OHLC").encode()))
    n = len(close)
    wiggle = np.abs(rng.normal(0, 0.16 * step, n))
    high = close + wiggle
    low = close - wiggle
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum.reduce([high, open_, close])
    low = np.minimum.reduce([low, open_, close])

    # explicit wick on the sweep bar (counted back from the end)
    sweep_idx = n - 1 - 1 - 1 - 2 - 1      # holder(0) + choch(1) + reclaim(2) + sweep(1) + low bar
    low[sweep_idx] = min(low[sweep_idx], close[sweep_idx] - direction * 1.0 * step)
    high[sweep_idx] = max(high[sweep_idx], close[sweep_idx] + direction * 0.2 * step)

    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close,
         "volume": rng.integers(80, 900, n).astype(float)},
        index=pd.DatetimeIndex(index, tz="UTC"),
    )
    return df


def write_demo_fixtures(
    out_dir: Path,
    config: Config | None = None,
    bars: int = 900,
    counter_offset: float = 2.0,
    end: datetime | None = None,
) -> list[Path]:
    """Write MT4-formatted demo CSVs (XAUUSD long, BTCUSD short, EURUSD quiet).

    Both 1m and 5m exports are written so the demo has the same timeframes as a
    live terminal (1m triggers, resampled 5m/1h/4h).
    """
    config = config or Config()
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    plan = {"XAUUSD": 1, "BTCUSD": -1, "EURUSD": 1}
    for symbol, direction in plan.items():
        for tf, count in ((1, bars * 5), (5, bars)):
            frame = synth_frame(symbol, tf, count, direction, end)
            # convert UTC back to server time for a realistic MT4 export
            local = frame.copy()
            local.index = local.index + pd.Timedelta(hours=counter_offset)
            path = out_dir / config.filename_pattern.format(symbol=symbol, timeframe=tf)
            lines = ["Date,Time,Open,High,Low,Close,Volume"]
            for ts, row in local.iterrows():
                lines.append(
                    f"{ts:%Y.%m.%d},{ts:%H:%M},{row['open']:.6f},{row['high']:.6f},"
                    f"{row['low']:.6f},{row['close']:.6f},{int(row['volume'])}"
                )
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            written.append(path)
    return written
