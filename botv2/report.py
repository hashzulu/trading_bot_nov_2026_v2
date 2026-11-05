"""Report rendering: markdown for humans, JSON for automation."""
from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .analysis import ChecklistItem, Factor, Setup, SymbolAnalysis, tf_label
from .config import Config
from .journal import (
    REPORTED_TOP_RULE_STATS,
    TOP_RULES,
    JournalProfile,
    setup_history,
    top_lessons,
)
from .narrative import (
    build_invalidation,
    build_thesis,
    build_watch_text,
    headline,
    journal_notes,
    journal_setup_line,
    journal_thesis,
    journal_watching,
)

try:
    from zoneinfo import ZoneInfo
    BERLIN = ZoneInfo("Europe/Berlin")
except Exception:                                   # pragma: no cover - no tzdata
    BERLIN = timezone.utc

STATUS_ICON = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌", "MANUAL": "👀"}


def _cet_stamp(value: datetime | None) -> str:
    """Recommendation time in Berlin local time (CET/CEST)."""
    if value is None:
        value = datetime.now(timezone.utc)
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(BERLIN).strftime("%Y-%m-%d %H:%M %Z")


def _fmt(value: float, digits: int = 5) -> str:
    return f"{value:.{digits}f}"


def _px(value: float) -> str:
    """2 decimals for indices/metals, 5 for FX."""
    return f"{value:.2f}" if value >= 10 else f"{value:.5f}"


def _symbol_table(analyses: list[SymbolAnalysis], config: Config) -> str:
    rows = [
        "| Symbol | Price | Bias | Session | 4h | 1h | 15m | 5m |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for analysis in analyses:
        trend_cells = []
        for tf in (240, 60, 15, 5):
            view = analysis.views.get(tf)
            if view is None:
                trend_cells.append("—")
            else:
                arrow = "↑" if view.trend > 0 else "↓" if view.trend < 0 else "→"
                trend_cells.append(f"{arrow} {view.rsi_now:.0f}")
        rows.append(
            f"| {analysis.symbol} | {_fmt(analysis.price)} | {analysis.bias_label} | "
            f"{analysis.session} | " + " | ".join(trend_cells) + " |"
        )
    return "\n".join(rows)


def _rules_block(setup, config: Config) -> str:
    gate = "PASS" if setup.rule_count >= config.min_top_rules else "FAIL"
    pieces = [("✅ " if key in setup.rules_ticked else "❌ ") + label
              for key, label in TOP_RULES.items()]
    lines = [f"**Top rules {setup.rule_count}/{len(TOP_RULES)}** "
             f"(gate ≥{config.min_top_rules}: **{gate}**) — " + " · ".join(pieces)]
    if setup.rules_missing:
        missing = ", ".join(TOP_RULES.get(k, k) for k in setup.rules_missing)
        lines.append(f"_Missing top rules: {missing}._")
    lines.append(f"**1-min primary trigger:** {setup.trigger_kind} "
                 f"at {setup.trigger_level:.5f}")
    return "\n".join(lines)


def _factor_table(setup) -> str:
    rows = ["| Confluence | Pts | Detail |", "|---|---|---|"]
    for factor in setup.factors:
        rows.append(f"| {factor.name} | +{factor.points} | {factor.detail} |")
    rows.append(f"| **Total** | **{setup.score}/{setup.max_score}** | grade **{setup.grade}** |")
    return "\n".join(rows)


def _checklist_table(setup) -> str:
    rows = ["| Check | Status | Note |", "|---|---|---|"]
    for item in setup.checklist:
        rows.append(f"| {item.key.replace('_', ' ')} | {STATUS_ICON.get(item.status, '•')} {item.status} | {item.text} |")
    return "\n".join(rows)


def _levels_table(analysis: SymbolAnalysis, limit: int = 8) -> str:
    if not analysis.levels:
        return "_no clustered levels_"
    ordered = sorted(analysis.levels, key=lambda lv: abs(lv.dist_atr))[:limit]
    rows = ["| Level | Side | TF | Δ ATR | Δ % |", "|---|---|---|---|---|"]
    for level in ordered:
        pct = (level.price - analysis.price) / analysis.price * 100 if analysis.price else 0.0
        rows.append(
            f"| {_fmt(level.price)} | {level.side} | {'/'.join(tf_label(t) for t in level.tfs)} | "
            f"{level.dist_atr:+.1f} | {pct:+.2f}% |"
        )
    return "\n".join(rows)


def _journal_section(profile: JournalProfile, analyses: list[SymbolAnalysis]) -> list[str]:
    lines = ["## Journal calibration", ""]
    lines.append(
        f"Read {profile.trades} logged trades (win rate **{profile.win_rate * 100:.0f}%**, "
        f"P&L **{profile.total_pnl:+,.0f}**). Thresholds and weights in this report are bent "
        f"towards what actually worked in that sample."
    )
    lines.append("")
    if profile.rsi_zones:
        lines += ["**RSI zones you actually watched**", "",
                  "| Symbol | Side | Median | p25 | p75 | n |", "|---|---|---|---|---|---|"]
        for zone in profile.rsi_zones:
            lines.append(f"| {zone.symbol} | {zone.direction} | {zone.median:.0f} | "
                         f"{zone.p25:.0f} | {zone.p75:.0f} | {zone.n} |")
        lines.append("")
    if profile.rule_matrix:
        stats = REPORTED_TOP_RULE_STATS
        lines += ["**The rule matrix (your own tracking)**", "",
                  f"High-confluence cohort: **{stats['trades']} trades**, "
                  f"**{stats['win_rate'] * 100:.1f}% win**, "
                  f"**${stats['pnl']:+,.2f}**, PF **{stats['profit_factor']:.2f}**, "
                  f"avg win/loss **{stats['avg_win_loss']:.2f}**, "
                  f"{stats['longs']} longs ({stats['long_win_rate'] * 100:.1f}% win) / "
                  f"{stats['shorts']} shorts ({stats['short_win_rate'] * 100:.1f}% win).", ""]
        cells = {(c.top, c.sub): c for c in profile.rule_matrix}
        lines += ["| top ↓ / sub → | 0 | 1 | 2 |", "|---|---|---|---|"]
        for top in range(0, 5):
            row = [f"| **{top} top**"]
            for sub in range(0, 3):
                cell = cells.get((top, sub))
                row.append("—" if cell is None else f"{cell.n} · {cell.win_rate * 100:.0f}%")
            lines.append(" | ".join(row) + " |")
        lines += ["", "_Cells show trade count · win rate. The gate requires **≥2 top rules**; "
                  "3+ top rules is the A-grade cohort._", ""]

    if profile.rules:
        lines += ["**Account rules in force:** " + " · ".join(profile.rules), ""]
    lessons = top_lessons(profile)
    if lessons:
        lines += ["**What the data says about your edges and leaks**", ""]
        lines += [f"- {lesson}" for lesson in lessons]
        lines.append("")
    return lines


def _setup_history_line(setup, profile: JournalProfile | None) -> str:
    stats = setup_history(profile, [factor.name for factor in setup.factors])
    if not stats:
        return "_No matching history in the journal for this exact mix._"
    parts = [f"{stat.label}: {stat.win_rate * 100:.0f}% win over {stat.n} "
             f"({stat.pnl:+,.0f})" for stat in stats[:4]]
    return "**Your journal with these ingredients:** " + " · ".join(parts) + "."


def _compliance_block(setup, config: Config) -> list[str]:
    """Per-trade compliance checklist: top rules + the primary 1-min trigger."""
    from .journal import TOP_RULES

    lines = ["**COMPLIANCE**", ""]
    lines.append(f"- ✅ 1-min primary trigger: {setup.trigger_kind} "
                 f"at {setup.trigger_level:.5f}")
    for key, label in TOP_RULES.items():
        lines.append(f"- {'✅' if key in setup.rules_ticked else '❌'} {label}")
    statuses = [item.status for item in setup.checklist]
    counts = {name: statuses.count(name) for name in ("PASS", "WARN", "FAIL", "MANUAL")}
    gate = "PASS" if setup.rule_count >= config.min_top_rules else "FAIL"
    lines += ["", f"gate **{gate}** · rules **{setup.rule_count}/5** · checklist "
                  f"{counts['PASS']}✅ {counts['WARN']}⚠ {counts['FAIL']}❌ {counts['MANUAL']}👀"]
    return lines


def _journal_block(analysis: SymbolAnalysis, setup, config: Config) -> list[str]:
    if not analysis.journal:
        return ["**NOTES**", "", f"- {analysis.bias_reason}.",
                "- Not enough data for a journal-style read."]
    if setup is None:
        missing = list(analysis.watch.missing) if analysis.watch else []
        lines = ["**SETUP**", ""]
        lines += [f"- {item}" for item in missing] or ["- no clean structure yet"]
        lines += ["", "**WATCHING**", ""]
        lines += [f"- {item}" for item in journal_watching(analysis, setup)]
        lines += ["", "**THESIS**", "", journal_thesis(analysis, setup), "",
                  "**NOTES**", ""]
        lines += [f"- {item}" for item in journal_notes(analysis, setup, config)]
        return lines
    lines = ["**SETUP**", "", journal_setup_line(analysis, setup), "",
             "**WATCHING**", ""]
    lines += [f"- {item}" for item in journal_watching(analysis, setup)]
    lines += ["", "**THESIS**", "", journal_thesis(analysis, setup), "",
              "**NOTES**", ""]
    lines += [f"- {item}" for item in journal_notes(analysis, setup, config)]
    return lines


def _setup_section(analysis: SymbolAnalysis, config: Config,
                   profile: JournalProfile | None = None,
                   generated: datetime | None = None) -> str:
    setup = analysis.setup
    assert setup is not None
    size = setup.position_size(config.typical_risk_usd)
    trigger_age = ("locked at recommendation" if setup.locked_sig
                   else f"{setup.bars_since_trigger} bars ago")
    stamp = _cet_stamp(setup.recommended_at or generated)
    lines = [
        f"### {analysis.symbol} — {setup.direction_label} — TRADE RECOMMENDED "
        f"({setup.grade}-grade, {setup.score}/{setup.max_score}) — "
        f"recommended {stamp} — "
        f"entry {_px(setup.entry)} · SL {_px(setup.sl)} · TP {_px(setup.tp)} · "
        f"{setup.rr:.1f}R",
        "",
        *_journal_block(analysis, setup, config),
        "",
        *_compliance_block(setup, config),
        "",
        _factor_table(setup),
        "",
        "**Trade plan**",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Direction | {setup.direction_label} |",
        f"| Recommended | {stamp} |",
        f"| Entry | {_fmt(setup.entry)} ({setup.entry_note}) |",
        f"| Stop | {_fmt(setup.sl)} — {setup.risk_atr:.2f} ATR |",
        f"| Target | {_fmt(setup.tp)} — {setup.tp_note} |",
        f"| R:R | 1:{setup.rr:.2f} |",
        f"| Size at ${config.typical_risk_usd:.0f} risk | {size:.4f} units (lots for FX) |",
        f"| Trigger | {setup.trigger_kind} at {_fmt(setup.trigger_level)} "
        f"({tf_label(setup.trigger_tf)}, {trigger_age}) |",
        "",
        f"**Invalidation.** {setup.invalidation}",
        "",
        _checklist_table(setup),
        "",
        _setup_history_line(setup, profile),
        "",
        "**Levels around price**",
        "",
        _levels_table(analysis),
        "",
    ]
    return "\n".join(lines)


def _watch_section(analysis: SymbolAnalysis, config: Config,
                   profile: JournalProfile | None = None) -> str:
    title = f"### {analysis.symbol} — {analysis.bias_label} — no trade yet"
    lines = [
        title,
        "",
        *_journal_block(analysis, None, config),
        "",
    ]
    lines += ["**Levels around price**", "", _levels_table(analysis), ""]
    return "\n".join(lines)


def render_markdown(analyses: list[SymbolAnalysis], config: Config,
                    data_source: str = "", generated: datetime | None = None,
                    profile: JournalProfile | None = None) -> str:
    generated = generated or datetime.now(timezone.utc)
    setup_count = sum(1 for a in analyses if a.setup)
    lines = [
        "# Chart read",
        "",
        "## TL;DR",
        "",
    ]
    for analysis in analyses:
        lines.append(f"- {headline(analysis)}")
    lines += ["", "## Market context", "", _symbol_table(analyses, config), ""]

    if setup_count:
        lines += ["## Tradeable setups", ""]
        for analysis in analyses:
            if analysis.setup:
                lines.append(_setup_section(analysis, config, profile, generated))
    lines += ["## Watchlist", ""]
    for analysis in analyses:
        if not analysis.setup:
            lines.append(_watch_section(analysis, config))

    notes = sorted({note for analysis in analyses for note in analysis.notes})
    if notes:
        lines += ["**Data notes**", ""] + [f"- {note}" for note in notes] + [""]

    if profile is not None:
        lines += _journal_section(profile, analyses)

    symbols_with_charts = [a.symbol for a in analyses]
    if config.save_charts and symbols_with_charts:
        lines += ["", "## Charts", "",
                  "_1h / 5m / 1m panels. Diamonds = 1h swings, small triangles = 5m swings, "
                  "dashed lines = HTF levels, bands = un-swept POIs, BOS/CHoC and H&S are 1-min, "
                  "stars/engulfing/hammers mark interesting candles, `rev` = 1-min bullish "
                  "reversal continuation._", ""]
        for symbol in symbols_with_charts:
            lines.append(f"**{symbol}** — [1h](charts/{symbol}_1h.png) · "
                         f"[5m](charts/{symbol}_5m.png) · [1m](charts/{symbol}_1m.png)")
            lines.append("")
            for tf_label_ in ("1h", "5m", "1m"):
                lines.append(f"![{symbol} {tf_label_} chart](charts/{symbol}_{tf_label_}.png)")
                lines.append("")

    lines += ["---", ""]
    if (Path(config.out_dir) / REC_LOG_NAME).exists():
        lines += [f"**Recommendation log:** [{REC_LOG_NAME}]({REC_LOG_NAME}) "
                  f"(every recommendation, append-only)", ""]
    lines += [
        "### How this read is built",
        "",
        "1. MT4 OHLCV is parsed per symbol/timeframe; coarser frames are resampled from the finest file.",
        "2. Wilder RSI/ATR, real swings (4H/1H/5M/1M), 1m BOS (LH/HL), CHoCH, sweeps and RSI divergence are computed.",
        "3. Bias comes from 4h/1h structure; the trigger must occur on the entry timeframe within the last "
        f"{config.trigger_window} bars.",
        "4. A setup needs a structure break plus an RSI extreme, divergence or sweep, and scores "
        f"≥{config.min_score_b} ({config.min_score_a}+ for A-grade).",
        "",
        "**Journal rules enforced in the checklist:** no incomplete patterns, no reversal without divergence, "
        "no entry when price is already extended, stops capped at "
        f"{config.max_sl_atr} ATR, ≥{config.min_rr}R, HTF alignment, news guard.",
        "",
        "---",
        "",
        f"**Generated:** {generated:%Y-%m-%d %H:%M} UTC  ",
        f"**Data:** {data_source or 'MT4 CSV'}  ",
        f"**Setup:** {len(analyses)} symbol(s) analysed, {setup_count} tradeable setup(s)",
    ]
    return "\n".join(lines)


def render_json(analyses: list[SymbolAnalysis], config: Config,
                data_source: str = "", generated: datetime | None = None,
                profile: JournalProfile | None = None) -> dict:
    generated = generated or datetime.now(timezone.utc)
    payload = {
        "generated": generated.isoformat(),
        "data_source": data_source,
        "settings": {
            "entry_tf": config.entry_tf,
            "rsi_floor": config.rsi_floor,
            "rsi_ceiling": config.rsi_ceiling,
            "min_score_b": config.min_score_b,
            "max_sl_atr": config.max_sl_atr,
            "typical_risk_usd": config.typical_risk_usd,
        },
        "symbols": {},
    }
    if profile is not None:
        payload["journal"] = profile.to_dict()
    for analysis in analyses:
        view_data = {}
        for tf, view in analysis.views.items():
            event = view.last_trigger_event()
            view_data[str(tf)] = {
                "trend": view.trend,
                "rsi": round(view.rsi_now, 2),
                "atr": round(view.atr_now, 6),
                "bars": len(view.frame),
                "last_event": None if event is None else {
                    "kind": event.kind, "direction": event.direction,
                    "level": event.level, "label": event.label,
                    "bars_ago": view.last_idx - event.idx,
                },
                "divergences": [
                    {"direction": d.direction, "label": d.label,
                     "bars_ago": view.last_idx - d.idx}
                    for d in view.divergences[-3:]
                ],
                "sweeps": [
                    {"direction": s.direction, "level": s.level, "source": s.source,
                     "bars_ago": view.last_idx - s.idx}
                    for s in view.sweeps[-3:]
                ],
            }
        payload["symbols"][analysis.symbol] = {
            "price": analysis.price,
            "entry_tf": analysis.entry_tf,
            "bias": analysis.bias,
            "bias_label": analysis.bias_label,
            "bias_reason": analysis.bias_reason,
            "session": analysis.session,
            "atr_entry": analysis.atr_entry,
            "notes": analysis.notes,
            "timeframes": view_data,
            "levels": [
                {"price": lv.price, "side": lv.side, "tfs": list(lv.tfs),
                 "dist_atr": round(lv.dist_atr, 3), "sources": list(lv.sources)}
                for lv in analysis.levels
            ],
            "setup": None if analysis.setup is None else {
                **asdict(analysis.setup),
                "factors": [asdict(f) for f in analysis.setup.factors],
                "checklist": [asdict(c) for c in analysis.setup.checklist],
                "levels": [],
                "direction_label": analysis.setup.direction_label,
                "signature": analysis.setup.signature,
            },
            "watch": None if analysis.watch is None else {
                "direction": analysis.watch.direction,
                "missing": analysis.watch.missing,
                "text": analysis.watch.text,
            },
        }
    return payload


def prepare_narratives(analyses: list[SymbolAnalysis], config: Config) -> None:
    for analysis in analyses:
        if analysis.setup:
            analysis.setup.thesis = build_thesis(analysis, analysis.setup, config)
            analysis.setup.invalidation = build_invalidation(analysis, analysis.setup)


def write_reports(analyses: list[SymbolAnalysis], config: Config,
                  data_source: str = "", generated: datetime | None = None,
                  out_dir: Path | None = None,
                  profile: JournalProfile | None = None) -> tuple[Path, Path]:
    generated = generated or datetime.now(timezone.utc)
    out_dir = out_dir or config.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    prepare_narratives(analyses, config)
    markdown = render_markdown(analyses, config, data_source, generated, profile)
    payload = render_json(analyses, config, data_source, generated, profile)

    stamp = generated.strftime("%Y%m%d_%H%M")
    md_path = out_dir / f"chart_read_{stamp}.md"
    json_path = out_dir / f"chart_read_{stamp}.json"
    md_path.write_text(markdown, encoding="utf-8")
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    (out_dir / "latest.md").write_text(markdown, encoding="utf-8")
    (out_dir / "latest.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return md_path, json_path


def console_summary(analyses: list[SymbolAnalysis], config: Config) -> str:
    prepare_narratives(analyses, config)
    lines = []
    for analysis in analyses:
        lines.append(headline(analysis))
        if analysis.setup:
            setup = analysis.setup
            lines.append(f"  entry {setup.entry:.5f} | SL {setup.sl:.5f} | TP {setup.tp:.5f} "
                         f"| {setup.rr:.1f}R | {setup.entry_note}")
            failed = [c.key for c in setup.checklist if c.status in ("FAIL", "WARN")]
            if failed:
                lines.append(f"  flags: {', '.join(failed)}")
        elif analysis.watch:
            lines.append(f"  waiting on: {'; '.join(analysis.watch.missing[:3])}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# append-only recommendation log
# --------------------------------------------------------------------------- #

REC_LOG_NAME = "recommended_trades.md"


def ensure_recommendation_log(config: Config) -> Path:
    """Create the append-only log with its header if it does not exist yet."""
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / REC_LOG_NAME
    if not path.exists():
        path.write_text(
            "# Recommended trades\n\n"
            "Append-only log of every trade recommendation the engine produced. "
            "Newest entries at the bottom.\n\n---\n\n",
            encoding="utf-8",
        )
    return path


def _rec_key(analysis: SymbolAnalysis, setup) -> str:
    day = ""
    view = analysis.views.get(analysis.entry_tf)
    if view is None and analysis.views:
        view = next(iter(analysis.views.values()))
    if view is not None and len(view.frame) > 0:
        day = view.frame.index[-1].strftime("%Y-%m-%d")
    return "|".join([setup.symbol, setup.direction_label, setup.trigger_kind,
                     f"{setup.trigger_level:.5f}", day])


def append_recommendation(analysis: SymbolAnalysis, setup, config: Config,
                          generated: datetime | None = None) -> bool:
    """Append one recommendation to reports/recommended_trades.md (deduped)."""
    from .narrative import journal_thesis

    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / REC_LOG_NAME
    key = _rec_key(analysis, setup)
    marker = f"<!-- sig: {key} -->"
    if path.exists() and marker in path.read_text(encoding="utf-8"):
        return False

    generated = generated or datetime.now(timezone.utc)
    digits = 2 if setup.entry >= 10 else 5
    if not path.exists():
        path.write_text(
            "# Recommended trades\n\n"
            "Append-only log of every trade recommendation the engine produced. "
            "Newest entries at the bottom.\n\n---\n\n",
            encoding="utf-8",
        )

    thesis = journal_thesis(analysis, setup)
    ticked = ", ".join(setup.rules_ticked) or "none"
    missing = ", ".join(setup.rules_missing) or "none"
    entry = (
        f"## {generated:%Y-%m-%d %H:%M} UTC — {setup.symbol} "
        f"{setup.direction_label} ({setup.grade}-grade, {setup.rr:.1f}R)\n\n"
        f"- **Entry:** {setup.entry:.{digits}f} ({setup.entry_note})\n"
        f"- **Stop:** {setup.sl:.{digits}f} — {setup.risk_atr:.2f} ATR\n"
        f"- **Target:** {setup.tp:.{digits}f} — {setup.tp_note}\n"
        f"- **R:R:** 1:{setup.rr:.2f}\n"
        f"- **Score:** {setup.score}/{setup.max_score} · trigger {setup.trigger_kind} "
        f"at {setup.trigger_level:.{digits}f} ({setup.bars_since_trigger} bars ago)\n"
        f"- **Top rules:** {setup.rule_count}/5 — {ticked}"
        + (f" · missing: {missing}" if setup.rules_missing else "") + "\n"
        f"- **Invalidation:** {setup.invalidation}\n"
        f"- **Status:** recommended — awaiting execution review\n\n"
        f"**THESIS**\n\n{thesis}\n\n{marker}\n\n---\n\n"
    )
    with path.open("a", encoding="utf-8") as handle:
        handle.write(entry)
    return True


# --------------------------------------------------------------------------- #
# Active recommendation plans
#
# Once a setup is recommended its entry / SL / TP / R must not move. The live
# analysis keeps producing fresh levels every cycle, so we pin the plan the
# first time it is logged and re-apply it to every later report, chart and
# dashboard build until the trade is closed (TP / SL / expiry).
# --------------------------------------------------------------------------- #

ACTIVE_PLANS_NAME = "active_plans.json"
ACTIVE_PLAN_STATUSES = ("recommended", "open")


def _plans_path(config: Config) -> Path:
    return Path(config.out_dir) / ACTIVE_PLANS_NAME


def load_active_plans(config: Config) -> dict[str, dict]:
    path = _plans_path(config)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _save_active_plans(plans: dict[str, dict], config: Config) -> None:
    path = _plans_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plans, indent=2, default=str), encoding="utf-8")


def _plan_key(analysis: SymbolAnalysis, setup) -> str:
    return _rec_key(analysis, setup)


def recommendation_signature(analysis: SymbolAnalysis, setup) -> str:
    """Stable id for a recommendation: same trigger + day = same trade.

    Unlike ``Setup.signature`` (which includes the live entry and grade), this
    does not change when price drifts, so the alert loop can recognise a
    re-detection instead of announcing a "new" trade with moved levels.
    """
    return _rec_key(analysis, setup)


def _parse_plan_time(value) -> datetime | None:
    if not value:
        return None
    try:
        stamp = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
    return stamp if stamp.tzinfo else stamp.replace(tzinfo=timezone.utc)


def _plan_created(plan: dict) -> datetime | None:
    return _parse_plan_time(plan.get("created"))


def latest_recommendation_at(config: Config) -> datetime | None:
    """When the newest recommendation was logged (any status)."""
    created = [_plan_created(plan) for plan in load_active_plans(config).values()]
    stamps = [stamp for stamp in created if stamp is not None]
    return max(stamps) if stamps else None


def enforce_one_new_recommendation(analyses: list[SymbolAnalysis], config: Config) -> int:
    """Allow at most one brand-new recommendation per cooldown window.

    Existing (pinned) recommendations are untouched. A setup that is suppressed
    becomes a plain watch item, so the report, the alert and the tracker can
    never disagree about which trade is live.
    """
    cooldown = float(getattr(config, "recommendation_cooldown_minutes", 10.0) or 0.0)
    if cooldown <= 0:
        return 0
    fresh = [a for a in analyses if a.setup is not None and not a.setup.locked_sig]
    if not fresh:
        return 0
    last = latest_recommendation_at(config)
    now = datetime.now(timezone.utc)
    age_min = ((now - last).total_seconds() / 60.0) if last is not None else None
    allowed = last is None or age_min is None or age_min >= cooldown
    keeper = max(fresh, key=lambda a: (a.setup.score, a.setup.rr)) if allowed else None
    dropped = 0
    for analysis in fresh:
        if analysis is keeper:
            continue
        if keeper is not None:
            reason = "another setup scored higher this cycle"
        elif last is not None and age_min is not None:
            wait = max(0.0, cooldown - age_min)
            reason = f"waiting {wait:.0f} min after the {last:%H:%M} UTC recommendation"
        else:
            reason = "one new trade at a time"
        analysis.notes.append(f"no new trade — {reason}")
        analysis.setup = None
        dropped += 1
    return dropped


def suppress_duplicate_setups(analyses: list[SymbolAnalysis], config: Config) -> int:
    """Drop a fresh setup whose trigger/day is already in the tracker.

    The same trigger must not be announced again just because an older plan is
    still open; that trade is already tracked. It also cannot be recorded twice
    because the plan key is identical, so dropping it keeps the report, alerts
    and tracker in sync.
    """
    plans = load_active_plans(config)
    if not plans:
        return 0
    dropped = 0
    for analysis in analyses:
        setup = analysis.setup
        if setup is None or setup.locked_sig:
            continue
        if _plan_key(analysis, setup) in plans:
            analysis.notes.append(
                "same setup is already tracked — no duplicate recommendation")
            analysis.setup = None
            dropped += 1
    return dropped


def trade_snapshot(analysis: SymbolAnalysis, config: Config,
                   profile: JournalProfile | None = None,
                   generated: datetime | None = None) -> dict:
    """The report content for one trade, stored with the recommendation."""
    section = (_setup_section(analysis, config, profile, generated)
               if analysis.setup is not None
               else _watch_section(analysis, config, profile))
    return {
        "tldr": headline(analysis),
        "market_context": _symbol_table([analysis], config),
        "watchlist": section,
    }


def _plan_payload(analysis: SymbolAnalysis, setup, config: Config,
                  generated: datetime | None = None,
                  profile: JournalProfile | None = None) -> dict:
    generated = generated or datetime.now(timezone.utc)
    setup.recommended_at = setup.recommended_at or generated
    signal_view = (analysis.views.get(int(setup.trigger_tf))
                   or analysis.views.get(analysis.entry_tf))
    signal_bar = (signal_view.frame.index[-1].isoformat()
                  if signal_view is not None and len(signal_view.frame) else None)
    return {
        "sig": _rec_key(analysis, setup),
        "signal_bar": signal_bar,
        "symbol": setup.symbol,
        "direction": setup.direction,
        "direction_label": setup.direction_label,
        "created": generated.isoformat(timespec="seconds"),
        "status": "recommended",
        "entry": setup.entry,
        "entry_note": setup.entry_note,
        "sl": setup.sl,
        "tp": setup.tp,
        "tp_note": setup.tp_note,
        "rr": setup.rr,
        "risk_atr": setup.risk_atr,
        "risk_usd": config.typical_risk_usd,
        "grade": setup.grade,
        "score": setup.score,
        "max_score": setup.max_score,
        "trigger_tf": setup.trigger_tf,
        "trigger_kind": setup.trigger_kind,
        "trigger_level": setup.trigger_level,
        "rule_count": setup.rule_count,
        "rules_ticked": list(setup.rules_ticked),
        "rules_missing": list(setup.rules_missing),
        "factors": [asdict(factor) for factor in setup.factors],
        "checklist": [asdict(item) for item in setup.checklist],
        "snapshot": trade_snapshot(analysis, config, profile, generated),
    }


def record_active_plan(analysis: SymbolAnalysis, setup, config: Config,
                       generated: datetime | None = None,
                       profile: JournalProfile | None = None) -> bool:
    """Pin a new recommendation. Returns True when a new plan was written."""
    plans = load_active_plans(config)
    key = _plan_key(analysis, setup)
    if key in plans:
        return False
    # One open trade at a time. The live plan is auto-closed when price hits its
    # SL/TP (or when it expires), so this no longer waits for a manual mark.
    for plan in plans.values():
        if (plan.get("symbol") == setup.symbol
                and not plan.get("test")
                and plan.get("status") in ACTIVE_PLAN_STATUSES):
            return False
    plans[key] = _plan_payload(analysis, setup, config, generated, profile)
    _save_active_plans(plans, config)
    return True


def attach_trade_charts(analysis: SymbolAnalysis, setup, config: Config) -> dict[str, str]:
    """Render and store the 1m/5m/1h screenshots for a just-recorded plan."""
    if not getattr(config, "save_charts", True):
        return {}
    plans = load_active_plans(config)
    key = _plan_key(analysis, setup)
    plan = plans.get(key)
    if not isinstance(plan, dict):
        return {}
    entry_time = _parse_plan_time(plan.get("opened_at") or plan.get("created"))
    exit_time = _parse_plan_time(plan.get("closed_at"))
    try:
        from .charts import snapshot_trade_charts

        charts = snapshot_trade_charts(
            analysis, setup, key, config,
            entry_time=entry_time, exit_time=exit_time)
    except Exception:
        charts = {}
    plan["charts"] = charts
    _save_active_plans(plans, config)
    return charts


def _refresh_trade_charts(analysis: SymbolAnalysis, plan: dict, config: Config) -> None:
    """Re-render the trade screenshots once the outcome is known."""
    if not getattr(config, "save_charts", True):
        return
    original = analysis.setup
    try:
        from .charts import snapshot_trade_charts

        setup = _setup_from_plan(analysis, plan)
        key = plan.get("sig") or _plan_key(analysis, setup)
        plan["charts"] = snapshot_trade_charts(
            analysis, setup, key, config,
            entry_time=_parse_plan_time(plan.get("opened_at") or plan.get("created")),
            exit_time=_parse_plan_time(plan.get("closed_at")))
    except Exception:
        pass
    finally:
        analysis.setup = original


def _apply_plan(setup, plan: dict) -> None:
    setup.direction = int(plan.get("direction", setup.direction))
    setup.entry = float(plan["entry"])
    setup.entry_note = plan.get("entry_note", setup.entry_note)
    setup.sl = float(plan["sl"])
    setup.tp = float(plan["tp"])
    setup.tp_note = plan.get("tp_note", setup.tp_note)
    setup.rr = float(plan.get("rr", setup.rr))
    setup.risk_atr = float(plan.get("risk_atr", setup.risk_atr))
    setup.grade = plan.get("grade", setup.grade)
    setup.score = int(plan.get("score", setup.score))
    setup.max_score = int(plan.get("max_score", setup.max_score))
    setup.trigger_tf = int(plan.get("trigger_tf", setup.trigger_tf))
    setup.trigger_kind = plan.get("trigger_kind", setup.trigger_kind)
    setup.trigger_level = float(plan.get("trigger_level", setup.trigger_level))
    setup.rule_count = int(plan.get("rule_count", setup.rule_count))
    setup.rules_ticked = list(plan.get("rules_ticked", setup.rules_ticked))
    setup.rules_missing = list(plan.get("rules_missing", setup.rules_missing))
    setup.factors = [Factor(**factor) for factor in plan.get("factors", [])]
    setup.checklist = [ChecklistItem(**item) for item in plan.get("checklist", [])]
    setup.locked_sig = plan.get("sig", "")
    setup.recommended_at = _plan_created(plan) or setup.recommended_at


def _setup_from_plan(analysis: SymbolAnalysis, plan: dict) -> Setup:
    """Rebuild the original recommendation when live detection finds no setup."""
    return Setup(
        symbol=plan["symbol"],
        direction=int(plan["direction"]),
        grade=plan.get("grade", "B"),
        score=int(plan.get("score", 0)),
        max_score=int(plan.get("max_score", 0)),
        entry=float(plan["entry"]),
        entry_note=plan.get("entry_note", ""),
        sl=float(plan["sl"]),
        tp=float(plan["tp"]),
        tp_note=plan.get("tp_note", ""),
        rr=float(plan.get("rr", 0)),
        risk_atr=float(plan.get("risk_atr", 0)),
        trigger_tf=int(plan.get("trigger_tf", analysis.entry_tf)),
        trigger_kind=plan.get("trigger_kind", "structure break"),
        trigger_level=float(plan.get("trigger_level", plan["entry"])),
        bars_since_trigger=0,
        factors=[Factor(**factor) for factor in plan.get("factors", [])],
        thesis="",
        invalidation="",
        checklist=[ChecklistItem(**item) for item in plan.get("checklist", [])],
        levels=analysis.levels,
        session=analysis.session,
        rules_ticked=list(plan.get("rules_ticked", [])),
        rules_missing=list(plan.get("rules_missing", [])),
        rule_count=int(plan.get("rule_count", 0)),
        locked_sig=plan.get("sig", ""),
        recommended_at=_plan_created(plan),
    )


def set_plan_result(plan: dict, status: str, close_price: float | None = None) -> None:
    """Record the realised P&L for a closed plan (points and R)."""
    plan["status"] = status
    if status not in ("hit_tp", "hit_sl"):
        for key in ("close_price", "pnl_points", "pnl_r", "pnl_usd"):
            plan.pop(key, None)
        return
    entry = float(plan.get("entry", 0.0))
    sl = float(plan.get("sl", entry))
    tp = float(plan.get("tp", entry))
    direction = 1 if int(plan.get("direction", 1)) > 0 else -1
    price = float(close_price) if close_price is not None else (
        tp if status == "hit_tp" else sl)
    risk = abs(entry - sl)
    points = direction * (price - entry)
    r_multiple = (points / risk) if risk else None
    risk_usd = float(plan.get("risk_usd", 300.0) or 300.0)
    plan["close_price"] = price
    plan["pnl_points"] = points
    plan["pnl_r"] = r_multiple
    plan["pnl_usd"] = (r_multiple * risk_usd) if r_multiple is not None else None


def _evaluate_plan_status(plan: dict, analysis: SymbolAnalysis, config: Config) -> str:
    status = plan.get("status", "recommended")
    if status not in ACTIVE_PLAN_STATUSES:
        return status
    view = (analysis.views.get(int(plan.get("trigger_tf", analysis.entry_tf)))
            or analysis.views.get(analysis.entry_tf))
    if view is None or len(view.frame) == 0:
        return status
    try:
        created = datetime.fromisoformat(plan["created"])
    except (KeyError, ValueError):
        return status
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    direction = 1 if int(plan.get("direction", 1)) > 0 else -1
    entry_value = plan.get("entry")
    entry = float(entry_value) if entry_value is not None else None
    sl = float(plan["sl"])
    tp = float(plan["tp"])

    # The trade is recommended on a completed candle; fills are only measured on
    # the candles immediately AFTER that signal candle.
    tf_minutes = max(1, int(plan.get("trigger_tf", analysis.entry_tf) or 1))
    signal_dt: datetime | None = None
    stored_signal = plan.get("signal_bar")
    if stored_signal:
        try:
            signal_dt = datetime.fromisoformat(str(stored_signal))
        except ValueError:
            signal_dt = None
        if signal_dt is not None and signal_dt.tzinfo is None:
            signal_dt = signal_dt.replace(tzinfo=timezone.utc)
    if signal_dt is None:
        # legacy plan: last complete candle at generation time
        minute = (created.hour * 60 + created.minute) // tf_minutes * tf_minutes
        bar_start = created.replace(hour=minute // 60, minute=minute % 60,
                                    second=0, microsecond=0)
        signal_dt = bar_start - timedelta(minutes=tf_minutes)
    after = view.frame[view.frame.index > signal_dt]

    # 1) Fill: a recommendation becomes OPEN once price trades its entry.
    if entry is not None and status == "recommended" and not plan.get("opened_at"):
        filled = (after[after["high"] >= entry] if direction < 0
                  else after[after["low"] <= entry])
        if not len(filled):
            age_hours = (datetime.now(timezone.utc) - created).total_seconds() / 3600.0
            if age_hours > float(getattr(config, "active_plan_max_age_hours", 12.0)):
                return "expired"
            return "recommended"
        opened_at = filled.index[0]
        plan["status"] = "open"
        plan["opened_at"] = opened_at.isoformat()
        status = "open"

    # 2) Outcome: evaluate SL/TP from the fill bar (or the first tradeable bar).
    opened_at = plan.get("opened_at")
    if opened_at:
        try:
            fill_bar = datetime.fromisoformat(str(opened_at)).replace(second=0, microsecond=0)
            after = view.frame[view.frame.index >= fill_bar]
        except ValueError:
            pass
    if len(after):
        if direction < 0:
            hits = after[after["high"] >= sl]
            if len(hits):
                plan["closed_at"] = hits.index[0].isoformat()
                set_plan_result(plan, "hit_sl", sl)
                return "hit_sl"
            hits = after[after["low"] <= tp]
            if len(hits):
                plan["closed_at"] = hits.index[0].isoformat()
                set_plan_result(plan, "hit_tp", tp)
                return "hit_tp"
        else:
            hits = after[after["low"] <= sl]
            if len(hits):
                plan["closed_at"] = hits.index[0].isoformat()
                set_plan_result(plan, "hit_sl", sl)
                return "hit_sl"
            hits = after[after["high"] >= tp]
            if len(hits):
                plan["closed_at"] = hits.index[0].isoformat()
                set_plan_result(plan, "hit_tp", tp)
                return "hit_tp"
    age_hours = (datetime.now(timezone.utc) - created).total_seconds() / 3600.0
    if age_hours > float(getattr(config, "active_plan_max_age_hours", 12.0)):
        return "expired"
    return status


def apply_active_plans(analyses: list[SymbolAnalysis], config: Config) -> int:
    """Pin every live recommendation to its original levels.

    Must run right after analysis and before reports/charts are written.
    Returns the number of analyses that were pinned.
    """
    plans = load_active_plans(config)
    if not plans:
        return 0
    applied = 0
    changed = False
    for analysis in analyses:
        symbol_plans = [
            plan for plan in plans.values()
            if plan.get("symbol") == analysis.symbol and not plan.get("test")
        ]
        if not symbol_plans:
            continue
        # Evaluate every live plan for this symbol. A plan can hit TP/SL even on a
        # cycle where the live direction flipped and nothing gets pinned; skipping
        # that check was why stopped-out trades stayed "recommended" forever.
        live = [plan for plan in symbol_plans
                if plan.get("status") in ACTIVE_PLAN_STATUSES]
        for plan in live:
            before = plan.get("status")
            _evaluate_plan_status(plan, analysis, config)
            if plan.get("status") != before:
                changed = True
                if plan.get("status") in ("hit_tp", "hit_sl"):
                    _refresh_trade_charts(analysis, plan, config)
        # One open trade at a time: while a plan is live it owns the headline,
        # even if live detection has moved on to a different trigger. No new
        # recommendation is offered until this plan hits TP/SL (or expires).
        active = [plan for plan in symbol_plans
                  if plan.get("status") in ACTIVE_PLAN_STATUSES]
        if not active:
            continue
        plan = max(active, key=lambda item: item.get("created", ""))
        if analysis.setup is None:
            analysis.setup = _setup_from_plan(analysis, plan)
            analysis.watch = None
        else:
            _apply_plan(analysis.setup, plan)
        stamp = plan.get("created", "")[:16].replace("T", " ")
        analysis.notes.append(
            f"trade plan locked to the {stamp} UTC recommendation — entry/SL/TP do not move"
        )
        applied += 1
    if changed:
        _save_active_plans(plans, config)
    return applied
