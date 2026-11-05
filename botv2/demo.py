"""Demo section: swing-detection screenshots built from live/demo OHLCV."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config
from .data import parse_mt4_csv, resample, tf_label
from .features import (atr, bos_events as find_bos_events,
                       choch_events as find_choch_events, detect_ranges,
                       real_swings, resolve_range_event, rolling_atr)

DEMO_DIR = Path(__file__).resolve().parent.parent / "data" / "demo"
TFS = (60, 5, 1)
WINDOW_BARS = {60: 180, 5: 260, 1: 260}
META_VERSION = 17
BOS_SWING_K = 2


def _swing_source(config: Config) -> tuple[str, Path, str] | None:
    """Live XAUUSD 1m first, then any live export, then the demo fixture."""
    directories = [(Path(config.data_dir), "live MT4 export"),
                   (DEMO_DIR, "demo fixture")]
    for directory, kind in directories:
        if not directory.is_dir():
            continue
        preferred = directory / "XAUUSD_1.csv"
        if preferred.exists():
            return "XAUUSD", preferred, kind
        for timeframe in (1, 5):
            for path in sorted(directory.glob(f"*_{timeframe}.csv")):
                symbol = path.name.rsplit("_", 1)[0].upper()
                return symbol, path, kind
    return None


def _signature(source: Path, config: Config, kind: str) -> dict:
    params = {str(tf): config.real_swing_params.get(tf, {}) for tf in TFS}
    return {
        "version": META_VERSION,
        "source": str(source),
        "kind": kind,
        "mtime": int(source.stat().st_mtime // 300) * 300,
        "atr_period": config.atr_period,
        "params": params,
        "bos_k": BOS_SWING_K,
        "windows": {str(tf): WINDOW_BARS[tf] for tf in TFS},
        "ranges": {
            "params_by_tf": config.range_params_by_tf,
            "scan_bars": config.range_scan_bars,
            "sweep_tol_atr": config.range_sweep_tol_atr,
            "reclaim_bars": config.range_reclaim_bars,
            "break_close_atr": config.range_break_close_atr,
        },
    }


def _unswept(frame: pd.DataFrame, swing) -> bool:
    """A level is only still a swing if later candles have not traded through it."""
    after = frame.iloc[swing.idx + 1:]
    if len(after) == 0:
        return True
    if swing.kind == "H":
        return bool((after["high"] < swing.price).all())
    return bool((after["low"] > swing.price).all())


def _bos_events(frame: pd.DataFrame, k: int = BOS_SWING_K):
    """v2 1m BOS (shared engine implementation)."""
    return find_bos_events(frame, k=k)


def _choch_events(frame: pd.DataFrame, k: int = BOS_SWING_K):
    """v2 1m CHoCH (shared engine implementation)."""
    return find_choch_events(frame, k=k)


def _render(frame: pd.DataFrame, symbol: str, tf: int,
            swings, params: dict, path: Path, source_label: str,
            bos_events: list | None = None,
            choch_events: list | None = None) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bars = min(WINDOW_BARS.get(tf, 300), len(frame))
    d = frame.iloc[-bars:]
    x = np.arange(len(d))
    up = d["close"].to_numpy() >= d["open"].to_numpy()
    colors = np.where(up, "#26a69a", "#ef5350")

    fig, ax = plt.subplots(figsize=(13.2, 6.4), facecolor="#0f141b")
    ax.set_facecolor("#0f141b")
    span = float(d["high"].max() - d["low"].min()) or 1.0
    pad = span * 0.012
    ax.vlines(x, d["low"], d["high"], colors=colors, linewidth=0.8)
    # Thick centered line bodies survive browser downscaling; rectangles can
    # appear as only their left half at small display sizes.
    plot_points = max(1.0, ax.get_position().width
                      * fig.get_size_inches()[0] * 72.0)
    body_lw = max(0.8, (plot_points / max(len(d), 1)) * 0.72)
    body_low = np.minimum(d["open"], d["close"])
    body_high = np.maximum(d["open"], d["close"])
    body_low = np.minimum(body_low, body_high - span * 0.0008)
    ax.vlines(x, body_low, body_high, color=colors, linewidth=body_lw, zorder=3)

    positions = {ts: i for i, ts in enumerate(d.index)}
    # Swept levels are stale: drop both the marker and its line.
    visible = [s for s in swings if s.t in positions and _unswept(frame, s)]
    for swing in visible:
        i = positions[swing.t]
        high = swing.kind == "H"
        color = "#ef5350"
        ax.plot(i, swing.price + (pad if high else -pad),
                "v" if high else "^", color=color, markersize=8,
                markeredgecolor="#0f141b", markeredgewidth=0.6, zorder=6)
        ax.hlines(swing.price, i, len(d) - 1, colors=color,
                  linestyles=":", linewidth=0.7, alpha=0.55, zorder=3)

    # 1m BOS labels: break candle is where the close crosses the LH/HL.
    bos_visible = [e for e in (bos_events or []) if e.t in positions]
    digits = 2 if float(d["close"].iloc[-1]) >= 10 else 5
    y_min, y_max = float(d["low"].min()), float(d["high"].max())

    # Every break gets a dot; the latest three per direction get a text label
    # so dense historical breaks do not cover the price action.
    for event in bos_visible:
        i = positions[event.t]
        up = event.direction > 0
        color = "#ef5350" if up else "#26a69a"
        anchor = float(d["high"].iloc[i]) if up else float(d["low"].iloc[i])
        ax.plot(i, anchor, "o", color=color, markersize=3.5, zorder=7)

    latest = ([e for e in bos_visible if e.direction > 0][-3:]
              + [e for e in bos_visible if e.direction < 0][-3:])
    latest.sort(key=lambda event: event.idx)
    for slot, event in enumerate(latest):
        i = positions[event.t]
        up = event.direction > 0
        color = "#ef5350" if up else "#26a69a"
        anchor = float(d["high"].iloc[i]) if up else float(d["low"].iloc[i])
        offset = pad * (1.8 + 1.7 * (slot % 4))
        y_text = anchor + offset if up else anchor - offset
        label = (f"BOS↑ {event.label or 'LH'} {event.level:.{digits}f}" if up
                 else f"BOS↓ {event.label or 'HL'} {event.level:.{digits}f}")
        ha = "right" if i > len(d) - 12 else "left" if i < 8 else "center"
        ax.text(i, y_text, label, ha=ha, va="bottom" if up else "top",
                fontsize=6.8, color=color, zorder=8,
                bbox=dict(boxstyle="round,pad=0.15", facecolor="#0f141b",
                          edgecolor=color, linewidth=0.5, alpha=0.9))
        if up:
            y_max = max(y_max, y_text)
        else:
            y_min = min(y_min, y_text)
    # CHoCH events use the same candle annotation style, in purple.
    choch_visible = [e for e in (choch_events or []) if e.t in positions]
    for event in choch_visible:
        i = positions[event.t]
        up = event.direction > 0
        anchor = float(d["high"].iloc[i]) if up else float(d["low"].iloc[i])
        ax.plot(i, anchor, "D", color="#ba68c8", markersize=4.5,
                markeredgecolor="#0f141b", markeredgewidth=0.5, zorder=7)

    latest_choch = ([e for e in choch_visible if e.direction > 0][-2:]
                    + [e for e in choch_visible if e.direction < 0][-2:])
    latest_choch.sort(key=lambda event: event.idx)
    for slot, event in enumerate(latest_choch):
        i = positions[event.t]
        up = event.direction > 0
        anchor = float(d["high"].iloc[i]) if up else float(d["low"].iloc[i])
        offset = pad * (2.2 + 1.7 * (slot % 3))
        y_text = anchor + offset if up else anchor - offset
        label = (f"CHoCH↑ {event.label or 'HL'} {event.level:.{digits}f}" if up
                 else f"CHoCH↓ {event.label or 'LH'} {event.level:.{digits}f}")
        ha = "right" if i > len(d) - 12 else "left" if i < 8 else "center"
        ax.text(i, y_text, label, ha=ha, va="bottom" if up else "top",
                fontsize=6.8, color="#ba68c8", zorder=9,
                bbox=dict(boxstyle="round,pad=0.15", facecolor="#0f141b",
                          edgecolor="#ba68c8", linewidth=0.5, alpha=0.9))
        if up:
            y_max = max(y_max, y_text)
        else:
            y_min = min(y_min, y_text)

    if bos_visible or choch_visible:
        ax.set_ylim(y_min - pad * 1.5, y_max + pad * 2.5)

    n = params.get("n", "?")
    prom = params.get("prom", "?")
    iso = params.get("iso_band", "?")
    highs = sum(1 for s in visible if s.kind == "H")
    lows = sum(1 for s in visible if s.kind == "L")
    bos_up = sum(1 for e in bos_visible if e.direction > 0)
    bos_down = sum(1 for e in bos_visible if e.direction < 0)
    choch_up = sum(1 for e in choch_visible if e.direction > 0)
    choch_down = sum(1 for e in choch_visible if e.direction < 0)
    ax.set_title(
        f"{symbol} {tf_label(tf)} — real swings  ·  n={n}  ·  "
        f"prominence ≥{prom} ATR  ·  isolation >{iso} ATR",
        color="#e8eef6", fontsize=12.5, loc="left", pad=12,
    )
    bos_line = (f"  ·  BOS {bos_up}↑ / {bos_down}↓"
                f"  ·  CHoCH {choch_up}↑ / {choch_down}↓" if tf == 1 else "")
    ax.text(0.0, 1.01,
            f"{highs} swing highs · {lows} swing lows shown{bos_line}  ·  "
            f"source: {source_label}  ·  pivot must be strict (ties fail)",
            transform=ax.transAxes, color="#8d96a4", fontsize=8.5, va="bottom")

    step = max(1, len(d) // 12)
    ax.set_xticks(x[::step])
    ax.set_xticklabels([ts.strftime("%m-%d %H:%M") for ts in d.index[::step]],
                       rotation=30, ha="right", color="#8d96a4", fontsize=8)
    ax.tick_params(axis="y", colors="#8d96a4", labelsize=8)
    ax.grid(alpha=0.16, color="#2a3140")
    for spine in ax.spines.values():
        spine.set_color("#252b33")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, facecolor=fig.get_facecolor())
    plt.close(fig)
    return {"highs": highs, "lows": lows, "bars": int(len(d)),
            "bos_up": bos_up, "bos_down": bos_down,
            "choch_up": choch_up, "choch_down": choch_down}


def _render_range(frame: pd.DataFrame, symbol: str, kind: str,
                  boxes: list, cases: list, path: Path,
                  source_label: str, params: dict,
                  sweep_tol: float, reclaim_bars: int) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    d = frame
    x = np.arange(len(d))
    up = d["close"].to_numpy() >= d["open"].to_numpy()
    colors = np.where(up, "#26a69a", "#ef5350")
    fig, ax = plt.subplots(figsize=(13.2, 6.4), facecolor="#0f141b")
    ax.set_facecolor("#0f141b")
    span = float(d["high"].max() - d["low"].min()) or 1.0
    pad = span * 0.012
    ax.vlines(x, d["low"], d["high"], colors=colors, linewidth=0.8)
    plot_points = max(1.0, ax.get_position().width
                      * fig.get_size_inches()[0] * 72.0)
    body_lw = max(0.8, (plot_points / max(len(d), 1)) * 0.72)
    body_low = np.minimum(d["open"], d["close"])
    body_high = np.maximum(d["open"], d["close"])
    body_low = np.minimum(body_low, body_high - span * 0.0008)
    ax.vlines(x, body_low, body_high, color=colors, linewidth=body_lw, zorder=3)

    def pos(ts) -> int:
        return int(d.index.searchsorted(ts, side="right")) - 1

    visible_boxes = 0
    for box in boxes:
        if box.end_t < d.index[0] or box.start_t > d.index[-1]:
            continue
        x0 = max(0, pos(box.start_t))
        x1 = min(len(d) - 1, pos(box.end_t))
        if x1 < 0 or x0 >= len(d) or x1 <= x0:
            continue
        ax.add_patch(Rectangle((x0, box.low), x1 - x0, box.high - box.low,
                               facecolor="#ffb74d", edgecolor="#ffb74d",
                               linewidth=0.8, alpha=0.10, zorder=2))
        ax.text(x0, box.high, f" range {box.low:.2f}–{box.high:.2f}",
                fontsize=6, color="#ffb74d", va="bottom", ha="left", zorder=7)
        visible_boxes += 1

    event_counts = {"breakout": 0, "sweep": 0}
    for _box, resolved in cases:
        entry_ts = resolved.get("entry_ts")
        if entry_ts is None:
            continue
        i = pos(entry_ts)
        if i < 0 or i >= len(d):
            continue
        kind_i = resolved["type"]
        if kind_i not in event_counts:
            continue
        long_side = resolved["trade"] == "long"
        color = "#ffd54f" if kind_i == "breakout" else "#ff8a65"
        marker = "^" if long_side else "v"
        y = float(d["low"].iloc[i]) - pad if long_side else float(d["high"].iloc[i]) + pad
        label = (f"BREAK {'↑' if long_side else '↓'}" if kind_i == "breakout"
                 else f"SWEEP {'↑' if long_side else '↓'}")
        ax.plot(i, y, marker, color=color, markersize=9, markeredgecolor="#0f141b",
                markeredgewidth=0.5, zorder=8)
        ax.annotate(label, xy=(i, y), xytext=(0, -16 if long_side else 14),
                    textcoords="offset points", ha="center",
                    va="top" if long_side else "bottom", fontsize=6.5,
                    color="#0f141b", fontweight="bold", zorder=9,
                    bbox=dict(boxstyle="round,pad=0.16", facecolor=color,
                              edgecolor="none", alpha=0.95))
        event_counts[kind_i] += 1

    box = cases[0][0] if cases else None
    if box is not None:
        title = (f"{symbol} 1m — range {kind}  ·  box "
                 f"{box.low:.2f}–{box.high:.2f}  ·  {box.bars} bars")
    else:
        title = f"{symbol} 1m — range {kind}"
    ax.set_title(title, color="#e8eef6", fontsize=12.5, loc="left", pad=12)
    ax.text(0.0, 1.01,
            f"1m boxes: {params.get('min_bars')}–{params.get('max_bars')} bars · "
            f"height ≤{params.get('k_height')} ATR · ER ≤{params.get('er_max')} · "
            f"{params.get('min_touch')} touches · {params.get('tol_atr')} ATR tol · "
            f"{params.get('break_atr')} ATR break  ·  sweep {sweep_tol} ATR / "
            f"{reclaim_bars} bars  ·  source: {source_label}",
            transform=ax.transAxes, color="#8d96a4", fontsize=8.2, va="bottom")

    step = max(1, len(d) // 12)
    ax.set_xticks(x[::step])
    ax.set_xticklabels([ts.strftime("%m-%d %H:%M") for ts in d.index[::step]],
                       rotation=30, ha="right", color="#8d96a4", fontsize=8)
    ax.tick_params(axis="y", colors="#8d96a4", labelsize=8)
    ax.grid(alpha=0.16, color="#2a3140")
    for spine in ax.spines.values():
        spine.set_color("#252b33")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, facecolor=fig.get_facecolor())
    plt.close(fig)
    return {"boxes": visible_boxes, **event_counts, "bars": int(len(d))}


def _range_images(config: Config, frame: pd.DataFrame, symbol: str,
                  source_label: str, demo_out: Path) -> list[dict]:
    params = config.range_params_by_tf.get(1)
    if not params or len(frame) < 80:
        return []

    # Walk back in chunks until a resolved 1m range event is found.
    boxes: list = []
    cases: list = []
    chunk = 20000
    end = len(frame)
    while end > 0:
        start = max(0, end - chunk)
        segment = frame.iloc[start:end]
        if len(segment) < 80:
            break
        satr = rolling_atr(segment, config.atr_period)
        found_boxes = detect_ranges(segment, satr, **params)
        found_cases = []
        for box in found_boxes:
            resolved = resolve_range_event(
                segment, satr, box.high, box.low, box.end_idx + 1,
                max_wait=config.range_resolve_max_wait,
                sweep_tol_atr=config.range_sweep_tol_atr,
                reclaim_bars=config.range_reclaim_bars,
                break_close_atr=config.range_break_close_atr)
            if not resolved or resolved.get("type") not in ("breakout", "sweep"):
                continue
            resolved["entry_ts"] = segment.index[int(resolved["entry_bar"])]
            found_cases.append((box, resolved))
        if found_cases:
            boxes, cases = found_boxes, found_cases
            break
        end = start

    if not cases:
        return []

    items: list[dict] = []
    for kind in ("breakout", "sweep"):
        subset = [c for c in cases if c[1]["type"] == kind]
        if not subset:
            continue
        box, resolved = max(subset, key=lambda item: item[1]["entry_ts"])
        entry_pos = int(frame.index.searchsorted(resolved["entry_ts"], side="right")) - 1
        box_pos = int(frame.index.searchsorted(box.start_t, side="right")) - 1
        start = max(0, min(entry_pos - 170, box_pos - 25))
        end = min(len(frame), entry_pos + 70)
        window = frame.iloc[start:end]
        if len(window) < 20:
            continue
        visible_boxes = [b for b in boxes
                         if b.end_t >= window.index[0] and b.start_t <= window.index[-1]]
        visible_cases = [c for c in cases
                         if window.index[0] <= c[1]["entry_ts"] <= window.index[-1]]
        name = f"{symbol}_1m_range_{kind}.png"
        counts = _render_range(window, symbol, kind, visible_boxes,
                               visible_cases, demo_out / name, source_label,
                               params, config.range_sweep_tol_atr,
                               config.range_reclaim_bars)
        items.append({
            "tf": f"1m range · {kind}",
            "file": name,
            "params": {
                **params,
                "sweep_tol_atr": config.range_sweep_tol_atr,
                "reclaim_bars": config.range_reclaim_bars,
                "break_close_atr": config.range_break_close_atr,
            },
            "kind": "range",
            **counts,
        })
    return items


def ensure_demo_charts(config: Config, out_dir: Path | None = None) -> dict:
    """Generate (or reuse) the three demo swing screenshots.

    Returns the metadata payload used by the dashboard's Demo tab. A missing
    demo fixture returns an empty dict so the tab can show a hint instead.
    """
    source_info = _swing_source(config)
    if source_info is None:
        return {}
    symbol, source, source_kind = source_info
    out_dir = Path(out_dir or config.out_dir)
    demo_out = out_dir / "demo"
    meta_path = demo_out / "swings_meta.json"
    signature = _signature(source, config, source_kind)
    source_label = source.name if source_kind.startswith("live") else f"data/demo/{source.name}"

    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("signature") == signature:
                images = [demo_out / item["file"] for item in meta.get("timeframes", [])]
                images += [demo_out / item["file"] for item in meta.get("ranges", [])]
                if images and all(path.exists() for path in images):
                    return meta
        except (OSError, json.JSONDecodeError, KeyError, TypeError):
            pass

    frame, _ = parse_mt4_csv(source, config.server_utc_offset_hours,
                             config.auto_server_offset)
    items: list[dict] = []
    for tf in TFS:
        params = config.real_swing_params.get(tf)
        if params is None:
            continue
        series = resample(frame, tf) if tf > 1 else frame
        if len(series) < 30:
            continue
        swings = real_swings(series, atr(series, config.atr_period), **params)
        bos = _bos_events(series) if tf == 1 else []
        choch = _choch_events(series) if tf == 1 else []
        name = f"{symbol}_{tf_label(tf)}_swings.png"
        counts = _render(series, symbol, tf, swings, params, demo_out / name,
                         source_label, bos_events=bos, choch_events=choch)
        items.append({
            "tf": tf_label(tf),
            "file": name,
            "params": params,
            **counts,
        })

    range_items = _range_images(config, frame, symbol, source_label, demo_out)
    meta = {
        "signature": signature,
        "symbol": symbol,
        "source": source_label,
        "source_kind": source_kind,
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "timeframes": items,
        "ranges": range_items,
    }
    demo_out.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta
