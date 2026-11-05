"""Narrative layer: writes the setup thesis in the trader's own vocabulary.

Deterministic by default. If an OpenAI-compatible endpoint is configured
(AI_BASE_URL / AI_API_KEY / AI_MODEL) it can be used to polish the summary,
but the report never depends on it.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request

from .analysis import Factor, Setup, SymbolAnalysis, WatchItem
from .config import Config
from .data import tf_label


def _factor(prefix: str, factors: list[Factor]) -> Factor | None:
    for factor in factors:
        if factor.name.lower().startswith(prefix.lower()):
            return factor
    return None


def _zone_text(rsi_low: float | None, config: Config) -> str:
    if rsi_low is None:
        return "momentum extreme"
    deep = rsi_low <= config.rsi_deep_floor
    if deep:
        return f"deep floor ({rsi_low:.0f} — I rarely see this hold twice in a row)"
    return f"floor zone (RSI {rsi_low:.0f})"


def build_thesis(analysis: SymbolAnalysis, setup: Setup, config: Config) -> str:
    factors = setup.factors
    entry_tf = tf_label(setup.trigger_tf)
    div = _factor("RSI divergence", factors)
    sweep = _factor("Liquidity sweep", factors)
    htf = _factor("HTF swing", factors)
    struct = _factor("BOS", factors) or _factor("CHoCH", factors)
    rsi_factor = _factor("RSI floor", factors) or _factor("RSI ceiling", factors)

    direction = "upside" if setup.direction > 0 else "downside"
    side = "long" if setup.direction > 0 else "short"
    parts: list[str] = []

    rsi_low = None
    if rsi_factor and setup.direction > 0:
        try:
            rsi_low = float(rsi_factor.detail.split("of ")[1].split()[0])
        except (IndexError, ValueError):
            rsi_low = None
    zone = _zone_text(rsi_low, config) if rsi_low is not None else None

    head = f"{setup.trigger_kind} {side} — "
    if struct and zone:
        head += f"{entry_tf} RSI tapped the {zone} and structure broke {direction}"
    elif struct:
        head += f"{entry_tf} structure broke {direction}"
    parts.append(head + ".")

    if div:
        extreme = "low" if setup.direction > 0 else "high"
        parts.append(f"The divergence read {div.detail}, so the {entry_tf} {extreme} was losing momentum.")
    if sweep:
        parts.append(f"{sweep.detail} — that is the liquidity grab I wait for.")
    if htf:
        parts.append(f"The {htf.detail} kept the higher-timeframe read valid.")

    target_side = "above" if setup.direction > 0 else "below"
    parts.append(
        f"I want continuation to {target_side} {setup.tp:.5f} ({setup.tp_note}); "
        f"{setup.rr:.1f}R against a {setup.risk_atr:.1f} ATR stop."
    )
    parts.append(
        f"Invalidation: a close back {'below' if setup.direction > 0 else 'above'} "
        f"{setup.trigger_level:.5f} and especially through {setup.sl:.5f}."
    )
    return " ".join(parts)


def build_invalidation(analysis: SymbolAnalysis, setup: Setup) -> str:
    return (
        f"Stop at {setup.sl:.5f} ({setup.risk_distance:.5f} = {setup.risk_atr:.2f} ATR). "
        f"Ideally the {'low' if setup.direction > 0 else 'high'} of the sweep candle and the "
        f"broken level {setup.trigger_level:.5f} are defended; a full close back through "
        f"{setup.trigger_level:.5f} invalidates the structure break."
    )


def build_watch_text(analysis: SymbolAnalysis, watch: WatchItem, config: Config) -> str:
    return (
        f"{'Long' if watch.direction > 0 else 'Short'} context — no trigger yet. "
        f"The checklist is missing: " + "; ".join(watch.missing) + "."
    )


def _price(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:.2f}" if value >= 10 else f"{value:.5f}"


def headline(analysis: SymbolAnalysis) -> str:
    """One-line 'what I find interesting' for the report header."""
    setup = analysis.setup
    note = getattr(analysis, "strategy_note", "")
    note_tail = f" · {note}" if note else ""
    if setup:
        suffix = " — checklist flags, treat as watchlist" if setup.grade == "C" else ""
        return (
            f"{analysis.symbol}: {setup.direction_label} {setup.grade}-grade on "
            f"{tf_label(setup.trigger_tf)} @ {setup.entry:.5f} "
            f"(SL {setup.sl:.5f} / TP {setup.tp:.5f}, {setup.rr:.1f}R){suffix}{note_tail}"
        )
    j = analysis.journal or {}
    direction = j.get("direction") or 0
    if analysis.watch:
        bits: list[str] = []
        # Same values as charts._entry_levels (journal_snapshot uses the shared helper).
        long_level = j.get("entry_long_above")
        short_level = j.get("entry_short_below")
        retest = j.get("entry_retest")
        if direction < 0:
            if short_level is not None:
                bits.append(f"SHORT: short < {_price(short_level)} (1-min BOS↓ HL)")
            if retest is not None:
                bits.append(f"RETEST: {_price(retest)} (neutral)")
            if long_level is not None:
                bits.append(f"LONG: long > {_price(long_level)} (1-min BOS↑ LH)")
        elif direction > 0:
            if long_level is not None:
                bits.append(f"LONG: long > {_price(long_level)} (1-min BOS↑ LH)")
            if retest is not None:
                bits.append(f"RETEST: {_price(retest)} (neutral)")
            if short_level is not None:
                bits.append(f"SHORT: short < {_price(short_level)} (1-min BOS↓ HL)")
        else:
            if long_level is not None:
                bits.append(f"LONG: long > {_price(long_level)}")
            if short_level is not None:
                bits.append(f"SHORT: short < {_price(short_level)}")
        tail = (" · potential entries: " + " · ".join(bits)) if bits else ""
        return f"{analysis.symbol}: {analysis.bias_label} — watching, no trigger{tail}{note_tail}"
    return f"{analysis.symbol}: {analysis.bias_label} — no actionable read{note_tail}"


def enhance_with_llm(payload: dict, config: Config) -> str | None:
    """Optional polish via an OpenAI-compatible /chat/completions endpoint."""
    if not (config.ai_base_url and config.ai_api_key and config.ai_model):
        return None
    url = config.ai_base_url.rstrip("/") + "/chat/completions"
    body = {
        "model": config.ai_model,
        "temperature": 0.2,
        "max_tokens": 600,
        "messages": [
            {"role": "system", "content": (
                "You are a trading assistant. Rewrite the structured analysis into a short, "
                "direct briefing for a discretionary FX/metals trader. Keep every number exact. "
                "Do not invent new setups; do not give financial advice beyond the given plan."
            )},
            {"role": "user", "content": json.dumps(payload, default=str)},
        ],
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {config.ai_api_key}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=config.ai_timeout) as response:
            parsed = json.loads(response.read().decode("utf-8"))
        return parsed["choices"][0]["message"]["content"].strip()
    except (urllib.error.URLError, KeyError, json.JSONDecodeError, TimeoutError, OSError):
        return None


# --------------------------------------------------------------------------- #
# Journal-style sections: WATCHING / THESIS / NOTES
# --------------------------------------------------------------------------- #

_DIRECTION_WORD = {1: "upside", -1: "downside"}


def _side(direction: int) -> str:
    return "long" if direction > 0 else "short"


def _signal_tf_label(analysis) -> str:
    return f"{analysis.journal.get('rsi_timeframe', 5)}-min"


def journal_setup_line(analysis, setup) -> str:
    """One line in the journal's SETUP style."""
    j = analysis.journal
    direction = j.get("direction") or 0
    if setup is not None:
        kind = setup.trigger_kind.lower()
        if "h&s" in kind:
            pattern = f"{'bullish' if setup.direction > 0 else 'bearish'} 1-min H&S neckline break"
        elif "choch" in kind or "choc" in kind:
            pattern = f"{'bullish' if setup.direction > 0 else 'bearish'} 1-min CHoC"
        elif "bos" in kind:
            break_text = "above LH" if setup.direction > 0 else "below HL"
            pattern = (f"{'bullish' if setup.direction > 0 else 'bearish'} "
                       f"1-min BOS ({break_text})")
        elif "range breakout" in kind:
            pattern = (f"{'bullish' if setup.direction > 0 else 'bearish'} "
                       f"1-min range breakout")
        elif "range sweep" in kind:
            pattern = (f"{'bullish' if setup.direction > 0 else 'bearish'} "
                       f"1-min range sweep")
        elif "hl sweep" in kind or "lh sweep" in kind:
            pattern = (f"{'bullish' if setup.direction > 0 else 'bearish'} "
                       f"1-min {setup.trigger_kind}")
        else:
            pattern = f"{'bullish' if setup.direction > 0 else 'bearish'} 1-min structure break"
        return f"Entered on a {pattern}"
    if not direction:
        return "No directional setup — waiting for structure to resolve"
    pattern = j.get("hs_forming") or j.get("hs_complete")
    if pattern:
        return (f"No entry yet — waiting on a {'bullish' if direction > 0 else 'bearish'} "
                f"1-min H&S neckline break")
    return (f"No entry yet — waiting on a {'bullish' if direction > 0 else 'bearish'} "
            f"1-min BOS / CHoCH / range breakout / range sweep / HL-LH sweep")


def journal_watching(analysis, setup) -> list[str]:
    """WATCHING bullets, mirroring the trade journal's style."""
    j = analysis.journal
    direction = j.get("direction") or 0
    if not direction:
        lines = []
        price = j.get("price", analysis.price)
        levels = [lv for lv in analysis.levels]
        support = max((lv.price for lv in levels if lv.price < price), default=None)
        resistance = min((lv.price for lv in levels if lv.price > price), default=None)
        if resistance:
            lines.append(f"1-hour / 4-hour resistance at {resistance:.5f}")
        if support:
            lines.append(f"1-hour / 4-hour support at {support:.5f}")
        lines.append(f"5-min RSI at {j['rsi_now']:.1f} "
                     f"(floor {j['floor']:.0f} / ceiling {j['ceiling']:.0f})")
        lines.append("1-min BOS/CHoC — no clean structure yet")
        if j.get("ny_open"):
            lines.append("NY open")
        return lines

    long_side = direction > 0
    lines: list[str] = []
    rules = j.get("rules") or {}

    def mark(ok: bool) -> str:
        return "✅" if ok else "⏳"

    rsi_ok = bool(rules.get("watchingRsi"))
    div_ok = bool(rules.get("rsiDivergence")) or bool(j.get("divergences"))
    confirm_ok = bool(rules.get("rsiConfirm")) or bool(j.get("confirm"))

    if long_side:
        lines.append(f"{mark(rsi_ok)} {_signal_tf_label(analysis)} RSI floor at {j['floor']:.0f} "
                     f"(window low {j['rsi_min']:.1f}, now {j['rsi_now']:.1f})")
    else:
        lines.append(f"{mark(rsi_ok)} {_signal_tf_label(analysis)} RSI ceiling at {j['ceiling']:.0f} "
                     f"(window high {j['rsi_max']:.1f}, now {j['rsi_now']:.1f})")

    if j.get("divergences"):
        labels = ", ".join(d["label"] for d in j["divergences"])
        lines.append(f"{mark(div_ok)} 5-min RSI divergence ({labels})")
    else:
        lines.append(f"{mark(div_ok)} 5-min RSI divergence (none yet)")
    lines.append(f"{mark(confirm_ok)} 5-min RSI confirmation after divergence "
                 f"({'yes' if confirm_ok else 'not yet'})")

    swing_word = "low" if long_side else "high"
    # Show the actual state first, then the next live support level when waiting.
    if j.get("htf_swept") and j.get("htf_level"):
        lines.append(f"✅ 1-hour swing {swing_word} at {j['htf_level']:.5f} swept")
    elif j.get("htf_respected") and j.get("htf_respect_level"):
        lines.append(f"✅ 1-hour swing {swing_word} at "
                     f"{j['htf_respect_level']:.5f} respected")
    elif j.get("levels_h1"):
        lines.append(f"⏳ 1-hour swing {swing_word} at {j['levels_h1'][0]:.5f} waiting")

    if j.get("m5_sweep"):
        source = j["m5_sweep"]["source"]
        label = "range" if source == "range" else f"5-min swing {swing_word}"
        lines.append(f"✅ {label} at {j['m5_sweep']['level']:.5f} swept")
    elif j.get("m5_respected") and j.get("m5_respect_level"):
        lines.append(f"✅ 5-min swing {swing_word} at "
                     f"{j['m5_respect_level']:.5f} respected")
    elif j.get("levels_m5"):
        lines.append(f"⏳ 5-min swing {swing_word} at {j['levels_m5'][0]:.5f} waiting")

    trigger_ok = bool(j.get("trigger")) or bool(j.get("hs_complete"))
    if j.get("hs_forming"):
        lines.append(f"⏳ 1-min {'bullish' if long_side else 'bearish'} H&S forming "
                     f"(neckline {j['hs_forming']['neckline']:.5f})")
    elif j.get("hs_complete"):
        lines.append(f"✅ 1-min {'bullish' if long_side else 'bearish'} H&S neckline broken "
                     f"at {j['hs_complete']['neckline']:.5f}")
    elif j.get("trigger"):
        lines.append(f"{mark(trigger_ok)} 1-min {j['trigger']['kind']} at {j['trigger']['level']:.5f} "
                     f"({j['trigger']['bars']} bars ago)")
    else:
        lines.append("⏳ 1-min triggers (BOS / CHoCH / range breakout / "
                     "range sweep / HL-LH sweep) — not yet")

    if j.get("ny_open"):
        lines.append("NY open")
    return lines


def journal_thesis(analysis, setup) -> str:
    j = analysis.journal
    direction = j.get("direction") or 0
    if not direction:
        h1 = analysis.views.get(240) or analysis.views.get(60)
        m1 = analysis.views.get(60) or analysis.views.get(15)
        h_trend = "bearish" if h1 and h1.trend < 0 else "bullish" if h1 and h1.trend > 0 else "flat"
        m_trend = "bullish" if m1 and m1.trend > 0 else "bearish" if m1 and m1.trend < 0 else "flat"
        levels = analysis.levels
        price = j.get("price", analysis.price)
        support = max((lv.price for lv in levels if lv.price < price), default=None)
        resistance = min((lv.price for lv in levels if lv.price > price), default=None)
        parts = [f"the higher timeframe is {h_trend} while the intraday is {m_trend}",
                 "no directional edge until one side gives way"]
        tail = ""
        if resistance and support:
            tail = f" — a close above {resistance:.5f} or below {support:.5f} starts the next leg"
        sentence = "; ".join(parts) + tail + "."
        return sentence[0].upper() + sentence[1:]

    long_side = direction > 0
    parts: list[str] = []
    if long_side:
        parts.append(f"the {_signal_tf_label(analysis)} RSI tapping the {j['floor']:.0f} floor"
                     if j.get("rules", {}).get("watchingRsi")
                     else f"the {_signal_tf_label(analysis)} RSI floor at {j['floor']:.0f}")
    else:
        parts.append(f"the {_signal_tf_label(analysis)} RSI tapping the {j['ceiling']:.0f} ceiling"
                     if j.get("rules", {}).get("watchingRsi")
                     else f"the {_signal_tf_label(analysis)} RSI ceiling at {j['ceiling']:.0f}")

    if j.get("divergences"):
        parts.append(f"a 5-min RSI divergence ({j['divergences'][-1]['label']})")
    if j.get("confirm"):
        parts.append("a 5-min RSI confirmation after divergence")
    if j.get("htf_level"):
        parts.append(f"price {'sweeping' if j.get('htf_swept') else 'respecting'}"
                     f" the 1-hour swing at {j['htf_level']:.5f}")
    if j.get("m5_sweep"):
        parts.append(f"a 5-min swing {'range ' if j['m5_sweep']['source'] == 'range' else ''}"
                     f"sweep at {j['m5_sweep']['level']:.5f}")
    elif j.get("m5_respected") and j.get("m5_respect_level"):
        parts.append(f"the 5-min swing respecting {j['m5_respect_level']:.5f}")
    if j.get("hs_complete"):
        parts.append("a completed 1-min H&S neckline break")
    elif j.get("hs_forming"):
        parts.append(f"a 1-min H&S forming around neckline {j['hs_forming']['neckline']:.5f}")
    elif j.get("trigger"):
        parts.append(f"a 1-min {j['trigger']['kind']} at {j['trigger']['level']:.5f}")

    move = ("continuation" if analysis.bias_label.startswith(("LONG (trend)", "SHORT (trend)"))
            else "a reversal")
    side_word = _DIRECTION_WORD[direction]

    if setup is not None:
        target = f"{setup.tp:.5f} ({setup.tp_note}; {setup.rr:.1f}R)"
        sentence = " plus ".join(parts) + f" signals {move} to the {side_word} to {target}."
        return sentence[0].upper() + sentence[1:]
    missing = [m for m in (analysis.watch.missing if analysis.watch else [])][:3]
    target_bits = []
    if j.get("targets_m5"):
        target_bits.append(f"the 5-min level at {j['targets_m5'][0]:.5f}")
    if j.get("targets_h1"):
        target_bits.append(f"the 1-hour level at {j['targets_h1'][0]:.5f}")
    target = f" to {target_bits[0]}" if target_bits else ""
    sentence = " plus ".join(parts) + f" would signal {move} to the {side_word}{target}."
    return sentence[0].upper() + sentence[1:]


def journal_notes(analysis, setup, config) -> list[str]:
    j = analysis.journal
    direction = j.get("direction") or 0
    notes: list[str] = []
    htf = analysis.views.get(240) or analysis.views.get(60)
    htf_name = "4-hour" if 240 in analysis.views else "1-hour"
    htf_trend = ("up" if htf and htf.trend > 0 else "down" if htf and htf.trend < 0 else "flat")

    if not direction:
        notes.append("No trade — higher timeframe and intraday structure conflict.")
        notes.append("Wait for a clean 1-min BOS/CHoC aligned with the 4h/1h read before doing anything.")
        notes.append("Max 3 trades/day; stop after 2 consecutive losses or -2R.")
        return notes

    long_side = direction > 0
    bias = analysis.bias_label
    if "trend" in bias:
        notes.append(f"Continuation {_side(direction)} trade in a {htf_name} {htf_trend}trend.")
    elif "pullback" in bias:
        notes.append(f"Reversal {_side(direction)} trade from the intraday pullback "
                     f"inside the {htf_name} {htf_trend}trend.")
    else:
        notes.append(f"Counter-trend {_side(direction)} idea — only take it with the full checklist.")

    from .journal import TOP_RULES
    if setup is not None:
        rule_count = setup.rule_count
        ticked = [TOP_RULES.get(k, k) for k in setup.rules_ticked]
        missing = [TOP_RULES.get(k, k) for k in setup.rules_missing]
    else:
        rule_count = j.get("rule_count", 0)
        ticked = [TOP_RULES.get(k, k) for k, ok in j.get("rules", {}).items() if ok]
        missing = [TOP_RULES.get(k, k) for k, ok in j.get("rules", {}).items() if not ok]
    gate = "PASS" if rule_count >= config.min_top_rules else "FAIL"
    notes.append(f"Top rules {rule_count}/5 ({', '.join(ticked) or 'none'}); "
                 f"gate {gate}" + (f"; missing {', '.join(missing)}" if missing else "") + ".")

    if setup is not None:
        notes.append(
            f"Plan: entry {setup.entry:.5f} ({setup.entry_note}) | SL {setup.sl:.5f} "
            f"({setup.risk_atr:.1f} ATR) | TP {setup.tp:.5f} | {setup.rr:.1f}R. "
            f"Never widen the stop after entry."
        )
        notes.append(f"Invalidation: a close back through {setup.trigger_level:.5f} "
                     f"kills the structure break.")
    else:
        notes.append("No plan yet — trigger required before sizing anything.")

    if j.get("rule_count", 0) < config.min_top_rules:
        notes.append("Journal: 0-1 top rules is a losing cohort (-$10.9k) — do not force this one.")
    if not j.get("divergences") and setup is None:
        notes.append("No divergence on the 5-min yet; the journal's divergence entries win 55% vs 38%.")
    if j.get("ny_open"):
        notes.append("Inside the NY-open window — watch for the liquidity grab before the real move.")
    return notes
