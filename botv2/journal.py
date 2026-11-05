"""Journal mining: learn the trader's actual zones, patterns and leaks.

Reads the Trading Journal SQLite DB (trades + per-trade compliance flags) and
derives a small, evidence-backed profile that the analyser uses:

* per-symbol RSI floor/ceiling zones actually watched,
* win-rate / P&L by compliance flag (divergence, 1h swing swept, 1m H&S, ...),
* the recurring execution leaks (widened stops, incomplete patterns, ...),
* account rules (max trades/day, never widen SL, stop after -2R),
* chronological edge (#3+ trade of the day).

The profile is cached as JSON so a run never depends on the DB being present.
"""
from __future__ import annotations

import json
import re
import sqlite3
import statistics
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import Config

# --------------------------------------------------------------------------- #
# profile model
# --------------------------------------------------------------------------- #

FLAG_LABELS = {
    "watchingRsi": "watching a key 5-min RSI level",
    "rsiDivergence": "5-min RSI divergence",
    "rsiConfirm": "5-min RSI confirmation (after divergence)",
    "sweptSwing": "1-hour swing swept",
    "breachedSwing": "1-hour swing breached",
    "respectedSwing": "1-hour swing respected",
    "swept5minSwing": "5-min swing swept",
    "breached5minSwing": "5-min swing breached",
    "respected5minSwing": "5-min swing respected",
    "entryPattern1m__hs1m": "1-min head & shoulders",
    "entryPattern1m__choc1m": "1-min CHoC",
    "entryPattern1m__bos1m": "1-min BOS",
    "entryPattern1m__range1m": "1-min range break",
    "entryPattern1m__sweeprange1m": "1-min range sweep",
    "entryPattern1m__swept1m": "1-min swept level",
    "entryPattern1m__discretionary1m": "1-min discretionary",
    "breakout1m": "1-min breakout",
}


TOP_RULES: dict[str, str] = {
    "sweptSwing": "1h swing respected / swept",
    "min5RespectedSwept": "5-min swing respected / swept",
    "watchingRsi": "watching a key 5-min RSI level",
    "rsiDivergence": "5-min RSI divergence",
    "rsiConfirm": "5-min RSI confirmation (after divergence)",
}



# the trader's own measurement of the high-confluence cohort (top x sub matrix,
# cells 2+0, 2+1, 2+2, 3+0, 3+1, 3+2, 4+1)
REPORTED_TOP_RULE_STATS = {
    "trades": 145,
    "win_rate": 0.579,
    "pnl": 9870.98,
    "profit_factor": 1.95,
    "avg_win_loss": 1.39,
    "longs": 90,
    "long_win_rate": 0.622,
    "shorts": 55,
    "short_win_rate": 0.509,
    "days": 69,
}

# rows observed in the trader's matrix: (top rules, sub patterns) -> (n, win rate)
REPORTED_MATRIX: dict[tuple[int, int], tuple[int, float]] = {
    (0, 0): (37, 0.11), (0, 1): (8, 0.38), (0, 2): (1, 1.00),
    (1, 0): (40, 0.35), (1, 1): (38, 0.42), (1, 2): (2, 1.00),
    (2, 0): (47, 0.51), (2, 1): (55, 0.49), (2, 2): (3, 1.00),
    (3, 0): (11, 0.64), (3, 1): (25, 0.80), (3, 2): (2, 0.50),
    (4, 1): (2, 1.00),
}

_PATTERN_FLAGS = (
    "entryPattern1m__hs1m", "entryPattern1m__choc1m", "entryPattern1m__bos1m",
    "entryPattern1m__range1m", "entryPattern1m__sweeprange1m",
    "entryPattern1m__swept1m", "entryPattern1m__discretionary1m",
    "entryPattern1m__swept1min",
)

_SUB_FLAG_MAP = {
    "choc1m": ("entryPattern1m__choc1m", "choc1m"),
    "hs1m": ("entryPattern1m__hs1m", "hs1m"),
    "bos1m": ("entryPattern1m__bos1m", "bos1m"),
    "range1m": ("entryPattern1m__range1m", "range1m"),
    "sweeprange1m": ("entryPattern1m__sweeprange1m", "sweeprange1m"),
    "swept1m": ("entryPattern1m__swept1m", "swept1m"),
}


@dataclass
class RuleCountStat:
    rules: int
    n: int
    win_rate: float
    pnl: float
    profit_factor: float


@dataclass
class RuleMatrixCell:
    top: int
    sub: int
    n: int
    win_rate: float
    pnl: float
    profit_factor: float

@dataclass
class FactorStat:
    key: str
    label: str
    n: int
    win_rate: float          # 0..1
    pnl: float
    baseline_win_rate: float
    baseline_pnl: float

    @property
    def edge(self) -> float:
        return self.win_rate - self.baseline_win_rate


@dataclass
class MistakeStat:
    key: str
    label: str
    n: int
    win_rate: float
    pnl: float


@dataclass
class RsiZone:
    symbol: str
    direction: str           # "long" | "short"
    median: float
    p25: float
    p75: float
    n: int


@dataclass
class JournalProfile:
    source: str = ""
    generated: str = ""
    trades: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    factors: list[FactorStat] = field(default_factory=list)
    mistakes: list[MistakeStat] = field(default_factory=list)
    rsi_zones: list[RsiZone] = field(default_factory=list)
    rules: list[str] = field(default_factory=list)
    pretrade_rules: list[str] = field(default_factory=list)
    tag_stats: list[FactorStat] = field(default_factory=list)
    rule_count_stats: list[RuleCountStat] = field(default_factory=list)
    rule_matrix: list[RuleMatrixCell] = field(default_factory=list)
    sequence_win_rates: dict[str, float] = field(default_factory=dict)
    lessons: list[str] = field(default_factory=list)

    def zone(self, symbol: str, direction: str) -> RsiZone | None:
        for zone in self.rsi_zones:
            if zone.symbol == symbol.upper() and zone.direction == direction:
                return zone
        return None

    def factor(self, key: str) -> FactorStat | None:
        return next((f for f in self.factors if f.key == key), None)

    def mistake(self, key: str) -> MistakeStat | None:
        return next((m for m in self.mistakes if m.key == key), None)

    def to_dict(self) -> dict[str, Any]:
        return {
            **{k: v for k, v in asdict(self).items()
               if k not in ("factors", "mistakes", "rsi_zones", "tag_stats")},
            "factors": [asdict(f) for f in self.factors],
            "mistakes": [asdict(m) for m in self.mistakes],
            "rsi_zones": [asdict(z) for z in self.rsi_zones],
            "tag_stats": [asdict(t) for t in self.tag_stats],
            "rule_count_stats": [asdict(r) for r in self.rule_count_stats],
            "rule_matrix": [asdict(r) for r in self.rule_matrix],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "JournalProfile":
        return cls(
            source=data.get("source", ""),
            generated=data.get("generated", ""),
            trades=data.get("trades", 0),
            win_rate=data.get("win_rate", 0.0),
            total_pnl=data.get("total_pnl", 0.0),
            factors=[FactorStat(**f) for f in data.get("factors", [])],
            mistakes=[MistakeStat(**m) for m in data.get("mistakes", [])],
            rsi_zones=[RsiZone(**z) for z in data.get("rsi_zones", [])],
            rules=data.get("rules", []),
            pretrade_rules=data.get("pretrade_rules", []),
            tag_stats=[FactorStat(**t) for t in data.get("tag_stats", [])],
            rule_count_stats=[RuleCountStat(**r) for r in data.get("rule_count_stats", [])],
            rule_matrix=[RuleMatrixCell(**r) for r in data.get("rule_matrix", [])],
            sequence_win_rates=data.get("sequence_win_rates", {}),
            lessons=data.get("lessons", []),
        )


# --------------------------------------------------------------------------- #
# DB discovery / loading
# --------------------------------------------------------------------------- #

DB_NAMES = ("TradingJournalDB.sqlite", "tradingjournaldb.sqlite")


def find_database(config: Config | None = None) -> Path | None:
    import os
    env = os.getenv("JOURNAL_DB")
    if env:
        path = Path(env).expanduser()
        return path if path.exists() else None
    candidates: list[Path] = []
    if config is not None and getattr(config, "journal_db", None):
        candidates.append(Path(config.journal_db))
    root = Path(__file__).resolve().parent.parent          # trading_botv2/
    workspace = root.parent
    candidates += [workspace / name for name in DB_NAMES]
    candidates += [workspace / "trading-journal-import-runtime-fixed" / name for name in DB_NAMES]
    candidates += [Path.cwd() / name for name in DB_NAMES]
    # one level of subdirectories (journal repo checkouts)
    try:
        for child in workspace.iterdir():
            if child.is_dir():
                for name in DB_NAMES:
                    candidates.append(child / name)
    except OSError:
        pass
    for path in candidates:
        try:
            if path.exists():
                return path
        except OSError:
            continue
    return None


def _load_rows(db_path: Path) -> tuple[list[dict], list[dict]]:
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        trades = [dict(r) for r in con.execute("select * from trades where pnl is not null")]
        try:
            compliance = [dict(r) for r in con.execute("select * from journalPerTradeCompliance")]
        except sqlite3.Error:
            compliance = []
    finally:
        con.close()
    return trades, compliance


# --------------------------------------------------------------------------- #
# extraction helpers
# --------------------------------------------------------------------------- #

_FLOOR_RE = re.compile(r"(?:floor|support)\s*(?:at|of)?\s*(\d{2}(?:\.\d+)?)", re.I)
_CEIL_RE = re.compile(r"(?:ceiling|resistance|cap)\s*(?:at|of)?\s*(\d{2}(?:\.\d+)?)", re.I)
_RSI_RE = re.compile(r"(?:5-min|5min|5 min)\s*rsi[^\d]{0,24}(\d{2}(?:\.\d+)?)", re.I)
_LEVEL_PAIR_RE = re.compile(r"RSI\s*(?:level|floor|ceiling)?\s*(?:at|of)?\s*(\d{1,2})\s*(?:and|,|/|&)\s*(\d{1,2})", re.I)


def _rsi_zones(trades: list[dict]) -> list[RsiZone]:
    buckets: dict[tuple[str, str], list[float]] = {}
    for trade in trades:
        text = " ".join(str(trade.get(k) or "") for k in ("watching", "thesis", "setup"))
        symbol = (trade.get("symbol") or "").upper()
        direction = (trade.get("direction") or "").lower()
        if not symbol or not text:
            continue

        def add(direction_key: str, value: float) -> None:
            if 5 <= value <= 95:
                buckets.setdefault((symbol, direction_key), []).append(value)

        for match in _FLOOR_RE.finditer(text):
            add("long" if direction == "long" else "long", float(match.group(1)))
        for match in _CEIL_RE.finditer(text):
            add("short" if direction == "short" else "short", float(match.group(1)))
        for match in _RSI_RE.finditer(text):
            value = float(match.group(1))
            add("long" if direction == "long" else "short", value)
        for match in _LEVEL_PAIR_RE.finditer(text):
            low, high = sorted((float(match.group(1)), float(match.group(2))))
            if low >= 5:
                add("long", low)
            if high <= 95:
                add("short", high)

    zones: list[RsiZone] = []
    for (symbol, direction), values in sorted(buckets.items()):
        if len(values) < 3:
            continue
        quartiles = statistics.quantiles(values, n=4) if len(values) >= 4 else [min(values), statistics.median(values), max(values)]
        zones.append(RsiZone(
            symbol=symbol,
            direction=direction,
            median=round(statistics.median(values), 1),
            p25=round(quartiles[0], 1),
            p75=round(quartiles[2], 1),
            n=len(values),
        ))
    return zones


_MISTAKE_PATTERNS = {
    "widened_stop": (r"increas(?:ed|ing) SL|moved SL|widened|\+\+ ?SL|SL to -",
                     "widened / moved the stop"),
    "early_close": (r"clos(?:ed|ing) early|closed at less|missed out",
                    "closed too early / left R on the table"),
    "before_news": (r"before news|news at|2\.30 ?pm", "entered around news"),
    "incomplete_pattern": (r"incomplete|before (?:it|the) form|didn.t wait|never happened",
                           "entered an incomplete pattern"),
    "oversized": (r"oversiz|max loss|increased risk", "oversized risk"),
    "no_divergence": (r"no rsi divergence|without.{0,30}divergence|didn.t have divergence",
                      "entered without divergence"),
}


def _factor_and_mistake_stats(trades: list[dict], compliance: list[dict]) -> tuple[list[FactorStat], list[MistakeStat]]:
    by_id = {t.get("id"): t for t in trades}
    total = len(trades)
    if total == 0:
        return [], []
    base_wins = sum(1 for t in trades if (t.get("pnl") or 0) > 0)
    base_rate = base_wins / total
    base_pnl = sum(t.get("pnl") or 0 for t in trades)

    factors: list[FactorStat] = []
    flags = set()
    for row in compliance:
        for key, value in row.items():
            if key in ("tradeId", "accountId", "updatedAt"):
                continue
            try:
                as_int = int(value or 0)
            except (TypeError, ValueError):
                as_int = 0
            if as_int:
                flags.add(key)

    for key in sorted(flags):
        subset = []
        for row in compliance:
            try:
                on = int(row.get(key) or 0) == 1
            except (TypeError, ValueError):
                on = False
            if on:
                trade = by_id.get(row.get("tradeId"))
                if trade:
                    subset.append(trade)
        if len(subset) < 4:
            continue
        wins = sum(1 for t in subset if (t.get("pnl") or 0) > 0)
        factors.append(FactorStat(
            key=key,
            label=FLAG_LABELS.get(key, key.replace("entryPattern1m__", "1m ")),
            n=len(subset),
            win_rate=wins / len(subset),
            pnl=sum(t.get("pnl") or 0 for t in subset),
            baseline_win_rate=base_rate,
            baseline_pnl=base_pnl,
        ))
    factors.sort(key=lambda f: -f.edge)

    mistakes: list[MistakeStat] = []
    for key, (pattern, label) in _MISTAKE_PATTERNS.items():
        regex = re.compile(pattern, re.I)
        subset = [t for t in trades if regex.search(str(t.get("execution") or ""))]
        if len(subset) < 2:
            continue
        wins = sum(1 for t in subset if (t.get("pnl") or 0) > 0)
        mistakes.append(MistakeStat(key, label, len(subset), wins / len(subset),
                                    sum(t.get("pnl") or 0 for t in subset)))
    mistakes.sort(key=lambda m: m.pnl)
    return factors, mistakes


def _truthy(value) -> bool:
    return value in (1, "1", True, "true")


def row_sub_patterns(row: dict) -> list[str]:
    hits: list[str] = []
    for name, flags in _SUB_FLAG_MAP.items():
        if any(_truthy(row.get(flag)) for flag in flags):
            hits.append(name)
    return hits


def row_top_rules(row: dict) -> list[str]:
    hits: list[str] = []
    # 1-hour swing group: respected / swept / breached counts as one rule
    if any(_truthy(row.get(k)) for k in ("sweptSwing", "breachedSwing", "respectedSwing")):
        hits.append("sweptSwing")
    # 5-min swing group
    if any(_truthy(row.get(k)) for k in
           ("5minrespectedswept", "respected5minSwing", "swept5minSwing", "breached5minSwing")):
        hits.append("min5RespectedSwept")
    for key in ("watchingRsi", "rsiDivergence", "rsiConfirm"):
        if _truthy(row.get(key)):
            hits.append(key)
    return hits


def top_rule_count(row: dict) -> int:
    return len(row_top_rules(row))


def _rule_matrix(trades: list[dict], compliance: list[dict]) -> list[RuleMatrixCell]:
    by_id = {t.get("id"): t for t in trades}
    buckets: dict[tuple[int, int], list[dict]] = {}
    for row in compliance:
        trade = by_id.get(row.get("tradeId"))
        if not trade:
            continue
        top = top_rule_count(row)
        sub = len(row_sub_patterns(row))
        buckets.setdefault((top, sub), []).append(trade)

    cells: list[RuleMatrixCell] = []
    for (top, sub), subset in sorted(buckets.items()):
        wins = sum(1 for t in subset if (t.get("pnl") or 0) > 0)
        gross_profit = sum(t.get("pnl") or 0 for t in subset if (t.get("pnl") or 0) > 0)
        gross_loss = -sum(t.get("pnl") or 0 for t in subset if (t.get("pnl") or 0) < 0)
        cells.append(RuleMatrixCell(
            top=top, sub=sub, n=len(subset),
            win_rate=wins / len(subset),
            pnl=sum(t.get("pnl") or 0 for t in subset),
            profit_factor=(gross_profit / gross_loss) if gross_loss else float("inf"),
        ))
    return cells


def _rule_count_stats(trades: list[dict], compliance: list[dict]) -> list[RuleCountStat]:
    """Aggregate the matrix by top-rule count (sub patterns collapsed)."""
    by_id = {t.get("id"): t for t in trades}
    buckets: dict[int, list[dict]] = {}
    for row in compliance:
        trade = by_id.get(row.get("tradeId"))
        if not trade:
            continue
        buckets.setdefault(top_rule_count(row), []).append(trade)
    out: list[RuleCountStat] = []
    for rules in sorted(buckets):
        subset = buckets[rules]
        wins = sum(1 for t in subset if (t.get("pnl") or 0) > 0)
        gp = sum(t.get("pnl") or 0 for t in subset if (t.get("pnl") or 0) > 0)
        gl = -sum(t.get("pnl") or 0 for t in subset if (t.get("pnl") or 0) < 0)
        out.append(RuleCountStat(rules=rules, n=len(subset), win_rate=wins / len(subset),
                                 pnl=sum(t.get("pnl") or 0 for t in subset),
                                 profit_factor=(gp / gl) if gl else float("inf")))
    return out


def _tag_stats(trades: list[dict]) -> list[FactorStat]:
    total = len(trades)
    if not total:
        return []
    base = sum(1 for t in trades if (t.get("pnl") or 0) > 0) / total
    buckets: dict[str, list[dict]] = {}
    for trade in trades:
        try:
            tags = json.loads(trade.get("tags") or "[]")
        except (TypeError, json.JSONDecodeError):
            tags = []
        for tag in tags:
            buckets.setdefault(str(tag), []).append(trade)
    out = []
    for tag, subset in buckets.items():
        if len(subset) < 5:
            continue
        wins = sum(1 for t in subset if (t.get("pnl") or 0) > 0)
        out.append(FactorStat(tag, tag.replace("_", " "), len(subset), wins / len(subset),
                              sum(t.get("pnl") or 0 for t in subset), base,
                              sum(t.get("pnl") or 0 for t in trades)))
    out.sort(key=lambda f: -f.win_rate)
    return out


def _account_rules(db_path: Path) -> tuple[list[str], list[str]]:
    try:
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        rows = list(con.execute("select distinct rules, preTradeRules from accounts"))
        con.close()
    except sqlite3.Error:
        return [], []
    rules: set[str] = set()
    pretrade: set[str] = set()
    for raw_rules, raw_pre in rows:
        for raw, target in ((raw_rules, rules), (raw_pre, pretrade)):
            if not raw:
                continue
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                continue
            for item in parsed:
                label = (item.get("label") or "").strip()
                if label:
                    target.add(label)
    return sorted(rules, key=str.lower), sorted(pretrade, key=str.lower)


def _sequence_stats(trades: list[dict]) -> dict[str, float]:
    buckets: dict[int, list[dict]] = {}
    for trade in trades:
        try:
            number = int(trade.get("number") or 1)
        except (TypeError, ValueError):
            number = 1
        buckets.setdefault(number, []).append(trade)
    out: dict[str, float] = {}
    for number, subset in buckets.items():
        if len(subset) < 3:
            continue
        wins = sum(1 for t in subset if (t.get("pnl") or 0) > 0)
        out[f"trade_{number}"] = round(wins / len(subset), 3)
    return out


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #

CACHE_NAME = "journal_profile.json"


def build_profile(db_path: Path) -> JournalProfile:
    trades, compliance = _load_rows(db_path)
    if not trades:
        raise ValueError(f"{db_path.name}: no trades with P&L")
    wins = sum(1 for t in trades if (t.get("pnl") or 0) > 0)
    factors, mistakes = _factor_and_mistake_stats(trades, compliance)
    rules, pretrade = _account_rules(db_path)
    profile = JournalProfile(
        source=str(db_path),
        generated=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        trades=len(trades),
        win_rate=wins / len(trades),
        total_pnl=sum(t.get("pnl") or 0 for t in trades),
        factors=factors,
        mistakes=mistakes,
        rsi_zones=_rsi_zones(trades),
        rules=rules,
        pretrade_rules=pretrade,
        tag_stats=_tag_stats(trades),
        sequence_win_rates=_sequence_stats(trades),
        rule_count_stats=_rule_count_stats(trades, compliance),
        rule_matrix=_rule_matrix(trades, compliance),
    )
    profile.lessons = _lessons(profile)
    return profile


def _pct(value: float) -> str:
    return f"{value * 100:.0f}%"


def _lessons(profile: JournalProfile) -> list[str]:
    lessons: list[str] = []
    if profile.rule_count_stats:
        two_plus = [r for r in profile.rule_count_stats if r.rules >= 2]
        one_or_less = [r for r in profile.rule_count_stats if r.rules < 2]
        if two_plus:
            n = sum(r.n for r in two_plus)
            pnl = sum(r.pnl for r in two_plus)
            wins = sum(r.win_rate * r.n for r in two_plus)
            lessons.append(
                f"Rule gate — with >=2 pre-trade rules ticked: {n} trades, "
                f"{wins / n * 100:.0f}% win, ${pnl:+,.0f}. "
                f"With 0-1 rules: "
                f"{sum(r.n for r in one_or_less)} trades, "
                f"{sum(r.pnl for r in one_or_less):+,.0f}. "
                f"Never take a setup with fewer than 2."
            )
    for mistake in profile.mistakes:
        if mistake.pnl < 0 and mistake.n >= 3:
            lessons.append(
                f"Leak — {mistake.label}: {mistake.n} trades, {_pct(mistake.win_rate)} win, "
                f"${mistake.pnl:+,.0f}. Hard brake."
            )
    best = [f for f in profile.factors if f.edge >= 0.08 and f.n >= 5][:4]
    for factor in best:
        lessons.append(
            f"Edge — {factor.label}: {_pct(factor.win_rate)} win over {factor.n} trades "
            f"(${factor.pnl:+,.0f}) vs {_pct(factor.baseline_win_rate)} baseline."
        )
    weak = [f for f in profile.factors if f.edge <= -0.05 and f.n >= 6][:2]
    for factor in weak:
        lessons.append(
            f"Weak — {factor.label}: {_pct(factor.win_rate)} over {factor.n} trades "
            f"(${factor.pnl:+,.0f}); demand extra confirmation."
        )
    return lessons


def load_profile(config: Config, refresh: bool = False) -> JournalProfile | None:
    cache = Path(config.out_dir) / CACHE_NAME
    if not refresh and cache.exists():
        try:
            return JournalProfile.from_dict(json.loads(cache.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, TypeError, KeyError):
            pass
    db_path = find_database(config)
    if db_path is None:
        return None
    try:
        profile = build_profile(db_path)
    except (sqlite3.Error, ValueError, OSError):
        return None
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(profile.to_dict(), indent=2), encoding="utf-8")
    return profile


def apply_calibration(config: Config, profile: JournalProfile | None) -> None:
    """Bend the strategy thresholds toward what the journal actually shows."""
    if profile is None or not getattr(config, "journal_calibration", True):
        return
    for symbol in config.symbols:
        floor_zone = profile.zone(symbol, "long")
        ceil_zone = profile.zone(symbol, "short")
        if floor_zone and floor_zone.n >= 5:
            # use the zone the trader actually watched, keeping the configured value as floor
            config.symbol_rsi_floor[symbol] = round(min(config.rsi_floor, floor_zone.median), 1)
        if ceil_zone and ceil_zone.n >= 5:
            config.symbol_rsi_ceiling[symbol] = round(max(config.rsi_ceiling, ceil_zone.median), 1)


def render_lessons_markdown(profile: JournalProfile) -> str:
    lines = [
        "# Journal profile",
        "",
        f"Source: `{profile.source}`  ",
        f"Trades: **{profile.trades}** · win rate **{_pct(profile.win_rate)}** · "
        f"P&L **{profile.total_pnl:+,.2f}**",
        "",
        "## What works",
        "",
        "| Factor | n | Win rate | P&L | vs baseline |",
        "|---|---|---|---|---|",
    ]
    for factor in profile.factors[:8]:
        lines.append(f"| {factor.label} | {factor.n} | {_pct(factor.win_rate)} | "
                     f"{factor.pnl:+,.0f} | {factor.edge:+.0%} |")
    leaks = [m for m in profile.mistakes if m.pnl < 0]
    habits = [m for m in profile.mistakes if m.pnl >= 0]
    lines += ["", "## Leaks", "", "| Behaviour | n | Win rate | P&L |", "|---|---|---|---|"]
    for mistake in leaks:
        lines.append(f"| {mistake.label} | {mistake.n} | {_pct(mistake.win_rate)} | ${mistake.pnl:+,.0f} |")
    if habits:
        lines += ["", "## Habits that paid", "", "| Behaviour | n | Win rate | P&L |", "|---|---|---|---|"]
        for mistake in habits:
            lines.append(f"| {mistake.label} | {mistake.n} | {_pct(mistake.win_rate)} | ${mistake.pnl:+,.0f} |")

    lines += ["", "## RSI zones actually watched", "",
              "| Symbol | Side | Median | p25 | p75 | n |", "|---|---|---|---|---|---|"]
    for zone in profile.rsi_zones:
        lines.append(f"| {zone.symbol} | {zone.direction} | {zone.median:.0f} | "
                     f"{zone.p25:.0f} | {zone.p75:.0f} | {zone.n} |")

    if profile.rule_count_stats:
        lines += ["", "## The >=2 rule gate", "",
                  f"Your own measurement: **{REPORTED_TOP_RULE_STATS['trades']} trades with "
                  f">=2 top rules ticked** → {REPORTED_TOP_RULE_STATS['win_rate'] * 100:.1f}% win, "
                  f"${REPORTED_TOP_RULE_STATS['pnl']:+,.2f}, PF "
                  f"{REPORTED_TOP_RULE_STATS['profit_factor']:.2f}.", "",
                  "| Rules ticked | Trades | Win rate | P&L | PF |", "|---|---|---|---|---|"]
        for stat in profile.rule_count_stats:
            pf = "inf" if stat.profit_factor == float("inf") else f"{stat.profit_factor:.2f}"
            lines.append(f"| {stat.rules} | {stat.n} | {stat.win_rate * 100:.0f}% | "
                         f"${stat.pnl:+,.0f} | {pf} |")
        lines += ["", "**Top rules:** " + " · ".join(TOP_RULES.values()), ""]

    if profile.rules:
        lines += ["", "## Account rules in force", ""] + [f"- {rule}" for rule in profile.rules]
    if profile.pretrade_rules:
        lines += ["", f"## Pre-trade checks in your account templates ({len(profile.pretrade_rules)})", ""]
        lines += [f"- {rule}" for rule in profile.pretrade_rules]
    if profile.lessons:
        lines += ["", "## Lessons the report will keep repeating", ""] + \
                 [f"- {lesson}" for lesson in profile.lessons]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# wiring into the report
# --------------------------------------------------------------------------- #

FACTOR_TO_FLAGS: dict[str, tuple[str, ...]] = {
    "RSI floor": ("watchingRsi",),
    "RSI ceiling": ("watchingRsi",),
    "RSI divergence": ("rsiDivergence",),
    "H&S": ("entryPattern1m__hs1m",),
    "BOS": ("entryPattern1m__bos1m",),
    "CHoCH": ("entryPattern1m__choc1m",),
    "Liquidity sweep": ("entryPattern1m__sweeprange1m", "sweptSwing"),
    "Range sweep": ("entryPattern1m__sweeprange1m",),
    "HTF swing": ("sweptSwing", "breachedSwing"),
    "HTF alignment": ("rsiConfirm",),
}


def setup_history(profile: JournalProfile | None, factor_names: list[str]) -> list[FactorStat]:
    """Historical stats for the ingredients this setup is built from."""
    if profile is None:
        return []
    seen: set[str] = set()
    out: list[FactorStat] = []
    for name in factor_names:
        for prefix, flags in FACTOR_TO_FLAGS.items():
            if not name.startswith(prefix):
                continue
            for flag in flags:
                if flag in seen:
                    continue
                stat = profile.factor(flag)
                if stat:
                    seen.add(flag)
                    out.append(stat)
    out.sort(key=lambda f: -f.edge)
    return out


def top_lessons(profile: JournalProfile | None, limit: int = 3) -> list[str]:
    if profile is None:
        return []
    return profile.lessons[:limit]
