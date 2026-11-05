"""Annotated 1h / 5m / 1m charts.

Uses an ordinal (bar-index) x-axis so weekend gaps and sparse MT4 exports don't
stretch the panels. Each panel is trimmed to a recent time window first.
"""
from __future__ import annotations

import json
import math
import os
import re
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .analysis import SymbolAnalysis, TFView, potential_entries
from .config import Config
from .data import parse_mt4_csv, tf_label
from .features import atr, detect_candle_patterns, detect_reversal_continuation, rsi

# recent window per timeframe: (days, max bars)
WINDOW = {240: (30, 180), 60: (9, 190), 5: (2.5, 280), 1: (0.5, 300)}
# trade screenshots are one step tighter than the normal chart window
TRADE_WINDOW_BARS = {60: 120, 5: 220, 1: 260}
# how many most-recent un-swept swing highs/lows to draw per timeframe
SWING_LINE_LIMIT = 1


def _mpl():
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/botv2-mpl-cache")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    return plt, Rectangle


def _style(ax) -> None:
    ax.set_facecolor("#0f141b")
    for spine in ax.spines.values():
        spine.set_color("#2a3140")
    ax.tick_params(colors="#aeb6c2", labelsize=7)
    ax.grid(True, color="#1b2230", linewidth=0.5, zorder=0)


def _time_ticks(ax, index: pd.DatetimeIndex, n: int = 6, fmt: str = "%d %b %H:%M") -> None:
    if len(index) == 0:
        return
    positions = np.linspace(0, len(index) - 1, min(n, len(index))).astype(int)
    ax.set_xticks(positions)
    ax.set_xticklabels([index[p].strftime(fmt) for p in positions], fontsize=6.5)


def _candles(ax, frame: pd.DataFrame, Rectangle) -> None:
    n = len(frame)
    if n == 0:
        return
    x = np.arange(n)
    up = frame["close"].to_numpy() >= frame["open"].to_numpy()
    colors = ["#26a69a" if u else "#ef5350" for u in up]
    ax.vlines(x, frame["low"], frame["high"], color=colors, linewidth=0.55, zorder=2)
    span = float(frame["high"].max() - frame["low"].min()) or 1.0
    min_body = span * 0.0008
    # Draw bodies as thick centered vertical lines rather than rectangles:
    # a rectangle can lose one side when the PNG is downscaled in the browser.
    plot_points = max(1.0, ax.get_position().width
                      * ax.figure.get_size_inches()[0] * 72.0)
    body_lw = max(0.9, (plot_points / max(n, 1)) * 0.90)
    body_low = np.minimum(frame["open"], frame["close"])
    body_high = np.maximum(frame["open"], frame["close"])
    body_low = np.minimum(body_low, body_high - min_body)
    ax.vlines(x, body_low, body_high, color=colors, linewidth=body_lw, zorder=3)
    ax.set_xlim(-1, n)


def _where(frame: pd.DataFrame, t) -> int | None:
    pos = frame.index.searchsorted(t, side="left")
    if pos < len(frame) and frame.index[pos] == t:
        return int(pos)
    return None


def _unswept_indices(view: TFView) -> set[int]:
    """Indices of swings whose level no later candle has traded through."""
    return view.unswept


def _mark_swings(ax, frame, view: TFView, big: bool, limit: int | None = None,
                 color: str = "#ef5350") -> None:
    source = view.major_swings or view.structure.swings
    unswept = _unswept_indices(view)
    swings = [s for s in source
              if s.idx in unswept and frame.index[0] <= s.t <= frame.index[-1]]
    if limit:
        swings = swings[-limit:]
    def pos_at(t):
        i = int(frame.index.searchsorted(t, side="right")) - 1
        return i if 0 <= i < len(frame) else None

    highs = [(pos_at(s.t), s.price) for s in swings if s.kind == "H"]
    lows = [(pos_at(s.t), s.price) for s in swings if s.kind == "L"]
    highs = [(x, p) for x, p in highs if x is not None]
    lows = [(x, p) for x, p in lows if x is not None]
    if big:
        if highs:
            ax.scatter([x for x, _ in highs], [p for _, p in highs], marker="D", s=44,
                       color=color, edgecolor="#0f141b", linewidth=0.5, zorder=6)
        if lows:
            ax.scatter([x for x, _ in lows], [p for _, p in lows], marker="D", s=44,
                       color=color, edgecolor="#0f141b", linewidth=0.5, zorder=6)
    else:
        if highs:
            ax.scatter([x for x, _ in highs], [p for _, p in highs], marker="v", s=15,
                       color=color, alpha=0.95, zorder=5)
        if lows:
            ax.scatter([x for x, _ in lows], [p for _, p in lows], marker="^", s=15,
                       color=color, alpha=0.95, zorder=5)


def _levels(ax, frame, view: TFView, price: float, color: str, limit: int | None,
            style: str, tag: str, label: bool = True) -> None:
    source = view.major_swings or view.structure.swings
    unswept = _unswept_indices(view)
    swings = [s for s in source[-60:]
              if s.idx in unswept and frame.index[0] <= s.t <= frame.index[-1]]
    swings = sorted(swings, key=lambda s: abs(s.price - price))
    if limit:
        swings = swings[:limit]
    digits = 5 if price < 10 else 2
    for swing in swings:
        pos = int(frame.index.searchsorted(swing.t, side="right")) - 1
        if pos < 0 or pos >= len(frame):
            continue
        ax.hlines(swing.price, pos, len(frame) - 1, color=color, linewidth=0.6,
                  linestyle=style, alpha=0.45, zorder=4)
        if label:
            ax.text(0.995, swing.price, f"{tag} {swing.price:.{digits}f} ",
                    transform=ax.get_yaxis_transform(), fontsize=5.4, color=color,
                    va="center", ha="right", zorder=8,
                    bbox=dict(boxstyle="round,pad=0.1", facecolor="#0f141b",
                              edgecolor="none", alpha=0.6))


def _poi(ax, frame, view: TFView, atr_v: float, color: str, tag: str, limit: int = 2) -> None:
    found = 0
    source = view.major_swings or view.structure.swings
    for swing in reversed(source):
        if not (frame.index[0] <= swing.t <= frame.index[-1]):
            continue
        after = view.frame.iloc[swing.idx + 1:]
        if len(after) == 0:
            continue
        swept = (after["low"] <= swing.price).any() if swing.kind == "L" \
            else (after["high"] >= swing.price).any()
        if swept:
            continue
        ax.axhspan(swing.price - 0.12 * atr_v, swing.price + 0.12 * atr_v,
                   color=color, alpha=0.13, zorder=1)
        ax.text(0.006, swing.price, f"POI {tag}", transform=ax.get_yaxis_transform(),
                fontsize=5.6, color=color, va="center", ha="left", zorder=7)
        found += 1
        if found >= limit:
            return


def _plan_levels(ax, setup) -> None:
    for value, color, label in ((setup.entry, "#42a5f5", "E"),
                                (setup.sl, "#ef5350", "SL"),
                                (setup.tp, "#66bb6a", "TP")):
        ax.axhline(value, color=color, linewidth=1.6, zorder=9)
        ax.text(0.35, value, f"{label} {value:.5f}", transform=ax.get_yaxis_transform(),
                fontsize=6.3, color=color, va="bottom", ha="left", zorder=10,
                bbox=dict(boxstyle="round,pad=0.12", facecolor="#0f141b",
                          edgecolor="none", alpha=0.8))


def _entry_levels(ax, frame, view: TFView, setup, atr_v: float,
                  show_potential: bool = True) -> None:
    """Potential entry points: 1m BOS above LH / below HL, retest zone, plan."""
    from .analysis import potential_entries

    summary: list[str] = []
    if show_potential:
        pe = potential_entries(view, atr_v)
        if pe.long_above is not None:
            level = pe.long_above
            ax.axhline(level, color="#66bb6a", linewidth=0.9, linestyle=":", zorder=6)
            ax.text(0.025, level, f"ENTRY ↑ {level:.5f} BOS↑ LH", transform=ax.get_yaxis_transform(),
                    fontsize=6.0, color="#66bb6a", ha="left", va="bottom", zorder=9,
                    bbox=dict(boxstyle="round,pad=0.12", facecolor="#0f141b",
                              edgecolor="none", alpha=0.75))
            summary.append(f"long > {level:.5f}")
        if pe.short_below is not None:
            level = pe.short_below
            ax.axhline(level, color="#ef5350", linewidth=0.9, linestyle=":", zorder=6)
            ax.text(0.025, level, f"ENTRY ↓ {level:.5f} BOS↓ HL", transform=ax.get_yaxis_transform(),
                    fontsize=6.0, color="#ef5350", ha="left", va="top", zorder=9,
                    bbox=dict(boxstyle="round,pad=0.12", facecolor="#0f141b",
                              edgecolor="none", alpha=0.75))
            summary.append(f"short < {level:.5f}")

        if pe.retest is not None:
            event_level = pe.retest
            ax.axhspan(event_level - 0.08 * atr_v, event_level + 0.08 * atr_v,
                       color="#90caf9", alpha=0.15, zorder=1)
            ax.text(0.995, event_level, f"retest {event_level:.5f} ",
                    transform=ax.get_yaxis_transform(), fontsize=5.9, color="#90caf9",
                    ha="right", va="center", zorder=9,
                    bbox=dict(boxstyle="round,pad=0.12", facecolor="#0f141b",
                              edgecolor="none", alpha=0.7))
            summary.append(f"retest {event_level:.5f}")

    if setup is not None:
        _plan_levels(ax, setup)

    if summary:
        ax.text(0.996, 0.02, "potential entries: " + " · ".join(summary),
                transform=ax.transAxes, fontsize=6.2, color="#c5cae9", ha="right",
                va="bottom", zorder=10,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="#0f141b",
                          edgecolor="#2a3140", alpha=0.9))


def _structure_1m(ax, frame, view: TFView, setup) -> None:
    events = [e for e in view.trigger_events()
              if frame.index[0] <= e.t <= frame.index[-1]][-9:]
    for event in events:
        x = _where(frame, event.t)
        if x is None:
            continue
        bull = event.direction > 0
        if event.kind == "BOS":
            text = (f"BOS{'↑' if bull else '↓'} {event.label}"
                    if event.label else "BOS")
        else:
            text = (f"CHoCH{'↑' if bull else '↓'} {event.label}"
                    if event.label else "CHoC")
        ha = ("right" if x > len(frame) - 12
              else "left" if x < 8 else "center")
        ax.annotate(text,
                    xy=(x, event.price), xytext=(0, 12 if bull else -16),
                    textcoords="offset points", ha=ha, fontsize=6, color="white",
                    zorder=8,
                    bbox=dict(boxstyle="round,pad=0.16",
                              facecolor="#1b5e20" if bull else "#b71c1c",
                              edgecolor="none", alpha=0.9))

    # Range breakout / range sweep / HL-LH sweep triggers
    extra = []
    for event in view.range_breakouts:
        if frame.index[0] <= event.t <= frame.index[-1]:
            extra.append((event.t, event.direction, event.price,
                          "RNG BRK", "#ffd54f"))
    for sweep in view.sweeps:
        if (sweep.source == "range"
                and frame.index[0] <= sweep.t <= frame.index[-1]):
            extra.append((sweep.t, sweep.direction,
                          float(frame.loc[sweep.t, "close"]),
                          "RNG SWP", "#ff8a65"))
    for event in view.sweep_reversals:
        if frame.index[0] <= event.t <= frame.index[-1]:
            text = "HL SWP" if event.direction > 0 else "LH SWP"
            extra.append((event.t, event.direction, event.price,
                          text, "#4dd0e1"))
    for ts, direction, y, text, color in sorted(extra, key=lambda item: item[0])[-9:]:
        x = _where(frame, ts)
        if x is None:
            continue
        bull = direction > 0
        ax.annotate(text, xy=(x, y), xytext=(0, 10 if bull else -14),
                    textcoords="offset points", ha="center", fontsize=5.6,
                    color="#0f141b", fontweight="bold", zorder=9,
                    bbox=dict(boxstyle="round,pad=0.14", facecolor=color,
                              edgecolor="none", alpha=0.92))

    for pattern in view.hs[-2:]:
        xs = [_where(frame, view.frame.index[i])
              for i in (pattern.left_idx, pattern.head_idx, pattern.right_idx)]
        if any(x is None for x in xs):
            continue
        ys = [pattern.left_price, pattern.head_price, pattern.right_price]
        ax.plot(xs, ys, color="#ce93d8", linewidth=0.9, marker="o", markersize=2.5,
                alpha=0.8, zorder=7)
        neck_x = xs[2] if not pattern.complete else (_where(
            frame, view.frame.index[pattern.break_idx]) or xs[2])
        ax.plot([xs[0], neck_x], [pattern.neckline, pattern.neckline],
                color="#ba68c8", linewidth=0.9,
                linestyle="-" if pattern.complete else "--", alpha=0.9, zorder=7)
        ax.text(xs[2], pattern.head_price,
                "H&S done" if pattern.complete else "H&S forming", fontsize=6,
                color="#e1bee7", ha="center",
                va="bottom" if pattern.direction > 0 else "top", zorder=9)


def _candles_marks(ax, frame, patterns, reversals, sweep_idxs) -> None:
    for pattern in patterns:
        if pattern.idx >= len(frame):
            continue
        y = float(frame["low"].iloc[pattern.idx]) if pattern.direction > 0 \
            else float(frame["high"].iloc[pattern.idx])
        marker = "^" if pattern.direction > 0 else "v"
        color = "#ffd54f" if pattern.direction > 0 else "#ff8a65"
        if pattern.kind == "capitulation":
            marker, color = "*", "#ffeb3b"
        elif pattern.kind.endswith("engulf"):
            color = "#80cbc4" if pattern.direction > 0 else "#ef9a9a"
        ax.scatter([pattern.idx], [y], marker=marker, s=34, color=color,
                   edgecolor="#0f141b", linewidth=0.4, zorder=8)
    for idx in reversals:
        if idx >= len(frame):
            continue
        after_sweep = any(0 <= idx - s_idx <= 3 for s_idx in sweep_idxs)
        ax.annotate("E/rev" if after_sweep else "rev",
                    xy=(idx, float(frame["low"].iloc[idx])), xytext=(0, -14),
                    textcoords="offset points", ha="center", fontsize=5.4,
                    color="#66bb6a" if after_sweep else "#a5d6a7", zorder=9,
                    bbox=dict(boxstyle="round,pad=0.1", facecolor="#0f141b",
                              edgecolor="none", alpha=0.65) if after_sweep else None)


def _window(view: TFView, tf: int) -> pd.DataFrame:
    days, bars = WINDOW.get(tf, (5, 300))
    frame = view.frame
    cut = frame.index[-1] - pd.Timedelta(days=days)
    tied = frame[frame.index >= cut].tail(bars)
    return tied if len(tied) >= 30 else frame.iloc[-bars:]


def _bar_pos(frame: pd.DataFrame, t) -> int | None:
    if t is None or len(frame) == 0:
        return None
    pos = int(frame.index.searchsorted(t, side="right")) - 1
    return max(0, min(len(frame) - 1, pos))


def _position_tool(ax, frame, setup, entry_time, exit_time, Rectangle,
                   show_labels: bool = True) -> None:
    """TradingView-style long/short position box from entry to exit."""
    if setup is None or entry_time is None:
        return
    x0 = _bar_pos(frame, entry_time)
    x1 = _bar_pos(frame, exit_time) if exit_time is not None else len(frame) - 1
    if x0 is None:
        return
    if x1 is None:
        x1 = len(frame) - 1
    if x1 < x0:
        x0, x1 = x1, x0
    if x1 <= x0:
        x1 = x0 + 3
    width = float(x1 - x0)
    entry = float(setup.entry)
    sl = float(setup.sl)
    tp = float(setup.tp)
    zones = [(min(entry, tp), max(entry, tp), "#26a69a"),
             (min(entry, sl), max(entry, sl), "#ef5350")]
    for bottom, top, color in zones:
        if top <= bottom:
            continue
        ax.add_patch(Rectangle((x0, bottom), width, top - bottom,
                               facecolor=color, edgecolor=color, alpha=0.13,
                               linewidth=0.7, zorder=5))
    digits = 2 if entry >= 10 else 5
    for value, color, label in ((entry, "#42a5f5", "entry"),
                                (sl, "#ef5350", "SL"),
                                (tp, "#66bb6a", "TP")):
        ax.plot([x0, x1], [value, value], color=color, linewidth=1.4, zorder=9)
        if show_labels:
            ax.text(0.995, value, f"{label} {value:.{digits}f}",
                    transform=ax.get_yaxis_transform(),
                    color=color, fontsize=6.2, va="center", ha="right", zorder=10,
                    bbox=dict(boxstyle="round,pad=0.12", facecolor="#0f141b",
                              edgecolor="none", alpha=0.8))
    risk = abs(entry - sl)
    rr = abs(tp - entry) / risk if risk else 0.0
    if show_labels:
        ax.text(x0 + width * 0.04, entry, f"{setup.direction_label} {rr:.1f}R",
                color="#e6ebf2", fontsize=6.0, va="bottom", ha="left", zorder=10)


def _rsi_panel(ax, frame, view: TFView, config: Config) -> None:
    _style(ax)
    values = view.rsi.reindex(frame.index)
    ax.plot(np.arange(len(frame)), values.to_numpy(),
            color="#ab47bc", linewidth=1.0)
    for level, color in ((config.rsi_ceiling, "#ef5350"), (70, "#7a3b3b"),
                         (50, "#4a5563"), (30, "#2e5c40"), (config.rsi_floor, "#66bb6a")):
        ax.axhline(level, color=color, linewidth=0.6, linestyle=":", alpha=0.9)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.yaxis.tick_right()
    ax.text(0.005, 0.94, "RSI 14", transform=ax.transAxes, fontsize=6.4,
            color="#ab47bc", va="top", ha="left")


def _figure(analysis: SymbolAnalysis, config: Config, tf: int,
            show_setup: bool = False, show_rsi: bool = False,
            trade_window: tuple[datetime, datetime] | None = None,
            entry_time: datetime | None = None,
            exit_time: datetime | None = None,
            position_tool: bool = False,
            right_margin: int = 0):
    """Build one price figure for a single timeframe."""
    plt, Rectangle = _mpl()
    views = analysis.views
    view = views[tf]
    if trade_window is not None:
        start, end = trade_window
        base = view.frame
        if end is not None:
            base = base[base.index <= end]
        if start is not None:
            base = base[base.index >= start]
        days, bars = WINDOW.get(tf, (5, 300))
        bars = TRADE_WINDOW_BARS.get(tf, bars)
        if len(base):
            cut = base.index[-1] - pd.Timedelta(days=days)
            tied = base[base.index >= cut].tail(bars)
            frame = tied if len(tied) >= 30 else base.iloc[-bars:]
        else:
            frame = base
        if (entry_time is not None and len(frame)
                and entry_time < frame.index[0]):
            frame = base[base.index >= entry_time - pd.Timedelta(minutes=tf * 2)]
        if len(frame) < 10:
            frame = _window(view, tf)
    else:
        frame = _window(view, tf)
    htf = views.get(60)
    mtf = views.get(5)
    entry = views.get(1) or views.get(5)

    if show_rsi:
        fig, (ax, ax_rsi) = plt.subplots(
            2, 1, figsize=(12.6, 7.2), squeeze=True, sharex=True,
            gridspec_kw={"height_ratios": [3.4, 1.0]})
    else:
        fig, ax = plt.subplots(1, 1, figsize=(12.6, 5.4), squeeze=True)
        ax_rsi = None
    fig.patch.set_facecolor("#0d1117")
    _style(ax)
    _candles(ax, frame, Rectangle)
    price = float(frame["close"].iloc[-1])

    # Primary: the current timeframe's own real swings, in red.
    _mark_swings(ax, frame, view, big=False, limit=None, color="#ef5350")
    _levels(ax, frame, view, price, "#ef5350", None, ":", tf_label(tf), label=True)
    _poi(ax, frame, view, view.atr_now, "#ef5350", tf_label(tf), limit=1)

    # Context: 1h swings in orange (skipped when this is the 1h chart).
    if htf is not None and tf != htf.tf:
        _mark_swings(ax, frame, htf, big=True, limit=140, color="#ffb74d")
        _levels(ax, frame, htf, price, "#ffb74d", None, "--", "1h",
                label=tf != 1)

    # Context: 5m swings are orange and labelled like 1h context swings.
    if mtf is not None and tf in (1, 60):
        _mark_swings(ax, frame, mtf, big=False, limit=260, color="#ffb74d")
        _levels(ax, frame, mtf, price, "#ffb74d", None, ":", "5m", label=True)

    if tf == (entry.tf if entry else None):
        atr_v = float(view.atr.iloc[-1]) or 1.0
        if htf is not None:
            _poi(ax, frame, htf, htf.atr_now, "#ffb74d", "1h", limit=1)
        if mtf is not None:
            _poi(ax, frame, mtf, mtf.atr_now, "#ffb74d", "5m", limit=1)
        _structure_1m(ax, frame, view, analysis.setup)
        if not position_tool:
            _entry_levels(ax, frame, view, analysis.setup, atr_v,
                          show_potential=analysis.setup is None)
        atr_cut = atr(view.frame).reindex(frame.index)
        pats = detect_candle_patterns(frame, atr_cut)
        revs = detect_reversal_continuation(frame, atr_cut, mtf.trend if mtf else 0)
        _candles_marks(ax, frame, pats, revs,
                       [sw.idx for sw in view.sweeps if sw.direction > 0])

    setup = analysis.setup
    entry_tf = entry.tf if entry else None
    if position_tool and setup is not None:
        _position_tool(ax, frame, setup, entry_time, exit_time, Rectangle,
                       show_labels=tf == 1)
    elif show_setup and setup is not None and tf != entry_tf:
        _plan_levels(ax, setup)
    if (position_tool or show_setup) and setup is not None:
        title = (f"{analysis.symbol} {tf_label(tf)} — "
                 f"{setup.direction_label} {setup.grade} ({setup.rr:.1f}R)")
    else:
        title = f"{analysis.symbol} {tf_label(tf)} — {analysis.bias_label}"
        title += (f" — {setup.direction_label} {setup.grade} ({setup.rr:.1f}R)"
                  if setup else " — no trigger")
    fig.suptitle(title, color="#e6ebf2", fontsize=13.5, x=0.008, y=0.985, ha="left")
    span = (frame.index[-1] - frame.index[0]).total_seconds() if len(frame) else 0
    fmt = "%H:%M" if span < 86400 else ("%d %b" if span > 5 * 86400 else "%d %b %H:%M")
    _time_ticks(ax, frame.index, n=6, fmt=fmt)
    if right_margin and len(frame):
        right_edge = len(frame) - 1
        ax.set_xlim(-0.5, right_edge + right_margin + 0.5)
    ax.yaxis.tick_left()
    ax.tick_params(axis="y", labelsize=7)
    if ax_rsi is not None:
        _rsi_panel(ax_rsi, frame, view, config)
        ax.tick_params(axis="x", labelbottom=False)
    legend_parts = [f"{tf_label(tf)} real swings = red"]
    if htf is not None and tf != htf.tf:
        legend_parts.append("1h context = orange")
    if mtf is not None and tf in (1, 60):
        legend_parts.append("5m context = orange")
    legend = "  ·  ".join(legend_parts) + "  ·  POI = un-swept zone"
    if tf == entry.tf:
        legend += ("  ·  BOS / CHoC / RNG BRK / RNG SWP / HL-LH SWP = 1m triggers"
                   "  ·  H&S = 1m pattern"
                   "  ·  ★ capitulation / ◆ engulfing / ^ hammer / v star"
                   "  ·  rev = bullish reversal continuation  ·  E/rev = after a sweep")
    fig.text(0.008, 0.012, legend, color="#7d8a9c", fontsize=7)
    if ax_rsi is not None:
        fig.subplots_adjust(left=0.05, right=0.97, top=0.92, bottom=0.08,
                            hspace=0.06)
    else:
        fig.subplots_adjust(left=0.05, right=0.97, top=0.90, bottom=0.12)
    return fig


def render_tf_chart(analysis: SymbolAnalysis, config: Config, tf: int,
                    out_dir: Path | None = None,
                    show_setup: bool = False,
                    show_rsi: bool = False,
                    trade_window: tuple[datetime, datetime] | None = None,
                    entry_time: datetime | None = None,
                    exit_time: datetime | None = None,
                    position_tool: bool = False,
                    right_margin: int = 0) -> Path | None:
    try:
        fig = _figure(analysis, config, tf, show_setup=show_setup,
                      show_rsi=show_rsi, trade_window=trade_window,
                      entry_time=entry_time, exit_time=exit_time,
                      position_tool=position_tool, right_margin=right_margin)
    except Exception:
        return None
    plt, _ = _mpl()
    out_dir = out_dir or config.chart_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{analysis.symbol}_{tf_label(tf)}.png"
    try:
        fig.savefig(path, facecolor=fig.get_facecolor(), dpi=125)
    except Exception:
        plt.close(fig)
        return None
    plt.close(fig)
    return path if path.exists() else None


def snapshot_trade_charts(analysis: SymbolAnalysis, setup, key: str,
                          config: Config,
                          entry_time: datetime | None = None,
                          exit_time: datetime | None = None) -> dict[str, str]:
    """Render 1m / 5m / 1h screenshots for one tracked trade."""
    slug = re.sub(r"[^A-Za-z0-9]+", "_", str(key)).strip("_")
    out_dir = Path(config.chart_dir) / "trades" / slug
    analysis.setup = setup
    last_bar = None
    for candidate in (1, 5, 60):
        view = analysis.views.get(candidate)
        if view is not None and len(view.frame):
            last_bar = view.frame.index[-1]
            break
    paths: dict[str, str] = {}
    right_bars = {1: 24, 5: 10, 60: 5}
    for tf in (1, 5, 60):
        if tf not in analysis.views:
            continue
        end = exit_time or last_bar
        if end is not None:
            end = end + timedelta(minutes=tf * right_bars.get(tf, 5))
        trade_window = (None, end) if end is not None else None
        path = render_tf_chart(
            analysis, config, tf, out_dir=out_dir, show_setup=True,
            show_rsi=True, trade_window=trade_window,
            entry_time=entry_time, exit_time=exit_time, position_tool=True,
            right_margin=right_bars.get(tf, 5))
        if path is not None:
            paths[tf_label(tf)] = f"/charts/trades/{slug}/{path.name}"
    return paths


def regenerate_plan_charts(plan: dict, config: Config) -> dict[str, str]:
    """Re-render a stored plan's trade screenshots from the price history."""
    from .analysis import analyse_symbol
    from .data import SymbolData, load_all, resample
    from .report import _parse_plan_time, _setup_from_plan

    data = {item.symbol: item for item in load_all(config).values()}
    symbol_data = data.get(plan.get("symbol"))
    if symbol_data is None:
        return {}
    base = symbol_data.frames.get(1)
    if base is None or len(base) < 60:
        return {}
    entry_time = _parse_plan_time(plan.get("opened_at") or plan.get("created"))
    exit_time = _parse_plan_time(plan.get("closed_at"))
    if exit_time is not None:
        base = base[base.index <= exit_time + timedelta(hours=12)]
    if len(base) < 60:
        return {}
    if (exit_time is not None and entry_time is not None
            and plan.get("status") in ("hit_tp", "hit_sl")):
        after = base[base.index >= entry_time]
        direction = int(plan.get("direction", 1))
        sl = float(plan.get("sl", 0.0))
        tp = float(plan.get("tp", 0.0))
        if plan.get("status") == "hit_sl":
            hits = after[after["low"] <= sl] if direction > 0 else after[after["high"] >= sl]
        else:
            hits = after[after["high"] >= tp] if direction > 0 else after[after["low"] <= tp]
        if len(hits):
            exit_time = hits.index[0]
    frames = {1: base}
    for tf in (5, 15, 60, 240):
        frames[tf] = resample(base, tf)
    analysis = analyse_symbol(
        SymbolData(plan.get("symbol"), frames, 1, derived={5, 15, 60, 240}), config)
    setup = _setup_from_plan(analysis, plan)
    key = plan.get("sig") or f"{plan.get('symbol')}_{plan.get('created')}"
    return snapshot_trade_charts(
        analysis, setup, key, config,
        entry_time=entry_time, exit_time=exit_time)


def render_symbol_charts(analysis: SymbolAnalysis, config: Config) -> list[Path]:
    """One PNG per timeframe: 4h, 1h, 5m, 1m."""
    out: list[Path] = []
    for tf in (240, 60, 5, 1):
        if tf not in analysis.views:
            continue
        path = render_tf_chart(analysis, config, tf)
        if path:
            out.append(path)
    return out


def render_all(analyses: list[SymbolAnalysis], config: Config) -> dict[str, list[Path]]:
    out: dict[str, list[Path]] = {}
    if not config.save_charts:
        return out
    for analysis in analyses:
        paths = render_symbol_charts(analysis, config)
        if paths:
            out[analysis.symbol] = paths
    return out


# --------------------------------------------------------------------------- #
# Lightweight Charts payload (smooth canvas zoom/pan, same engine as chart-app)
# --------------------------------------------------------------------------- #

def _un_swept_swings(view: TFView, frame, limit: int | None = None):
    unswept = _unswept_indices(view)
    swings = [s for s in (view.major_swings or view.structure.swings)
              if s.idx in unswept
              and frame.index[0] <= s.t <= frame.index[-1]]
    highs = sorted((s for s in swings if s.kind == "H"),
                   key=lambda s: s.idx, reverse=True)
    lows = sorted((s for s in swings if s.kind == "L"),
                  key=lambda s: s.idx, reverse=True)
    if limit:
        highs, lows = highs[:limit], lows[:limit]
    return highs, lows


def _swing_price_lines(view: TFView, frame, digits: int, color: str, tag: str,
                       limit: int | None = None) -> list[dict]:
    lines = []
    for side, points in zip(("high", "low"), _un_swept_swings(view, frame, limit)):
        for swing in points:
            lines.append({
                "price": float(swing.price), "color": color, "width": 1,
                "style": 0, "title": f"{tag} {side} {swing.price:.{digits}f}",
                "axis_label": True,
            })
    return lines


def _poi_bands(view: TFView, frame, atr_v: float, color: str, tag: str,
               limit: int = 1) -> list[dict]:
    bands = []
    for swing in reversed(view.major_swings or view.structure.swings):
        if not (frame.index[0] <= swing.t <= frame.index[-1]):
            continue
        after = view.frame.iloc[swing.idx + 1:]
        if len(after) == 0:
            continue
        swept = ((after["low"] <= swing.price).any() if swing.kind == "L"
                 else (after["high"] >= swing.price).any())
        if swept:
            continue
        bands.append({
            "y0": float(swing.price - 0.12 * atr_v),
            "y1": float(swing.price + 0.12 * atr_v),
            "color": color, "opacity": 0.13, "label": f"POI {tag}",
            "from_time": int(frame.index[0].timestamp()),
            "to_time": int(frame.index[-1].timestamp()),
        })
        if len(bands) >= limit:
            break
    return bands


def _swing_markers(view: TFView, frame, color: str, tag: str,
                   digits: int, big: bool = False) -> list[dict]:
    """Swing triangles/diamonds for the interactive chart."""
    out: list[dict] = []
    unswept = _unswept_indices(view)
    for swing in view.major_swings or view.structure.swings:
        if swing.idx not in unswept:
            continue
        if not (frame.index[0] <= swing.t <= frame.index[-1]):
            continue
        pos = int(frame.index.searchsorted(swing.t, side="right")) - 1
        if pos < 0 or pos >= len(frame):
            continue
        high = swing.kind == "H"
        out.append({
            "time": int(frame.index[pos].timestamp()),
            "price": float(swing.price),
            "position": "aboveBar" if high else "belowBar",
            "shape": "square" if big else ("arrowDown" if high else "arrowUp"),
            "color": color,
            "text": "",
            "tooltip": (f"{tag} swing {'high' if high else 'low'} "
                        f"{swing.price:.{digits}f}"),
        })
    return out


def _range_bands(view: TFView, frame, digits: int,
                 limit: int = 12) -> list[dict]:
    """Amber boxes for the detected 5m ranges."""
    bands: list[dict] = []
    for box in list(getattr(view, "ranges", []) or [])[-limit:]:
        if box.end_t < frame.index[0] or box.start_t > frame.index[-1]:
            continue
        start = max(box.start_t, frame.index[0])
        end = min(box.end_t, frame.index[-1])
        bands.append({
            "y0": float(box.low), "y1": float(box.high),
            "from_time": int(start.timestamp()),
            "to_time": int(end.timestamp()),
            "color": "#ffb74d", "opacity": 0.10,
            "label": f"range {box.low:.{digits}f}-{box.high:.{digits}f}",
        })
    return bands


def _range_markers(view: TFView, frame, digits: int) -> list[dict]:
    out: list[dict] = []
    for event in getattr(view, "range_breakouts", []) or []:
        if not (frame.index[0] <= event.t <= frame.index[-1]):
            continue
        bull = event.direction > 0
        out.append({
            "time": int(event.t.timestamp()), "price": float(event.price),
            "position": "aboveBar" if bull else "belowBar",
            "shape": "circle", "color": "#ffd54f", "text": "RNG BRK",
            "tooltip": f"range breakout {'up' if bull else 'down'} {event.level:.{digits}f}",
        })
    for sweep in view.sweeps:
        if sweep.source != "range" or not (frame.index[0] <= sweep.t <= frame.index[-1]):
            continue
        bull = sweep.direction > 0
        out.append({
            "time": int(sweep.t.timestamp()), "price": float(sweep.level),
            "position": "aboveBar" if bull else "belowBar",
            "shape": "circle", "color": "#ff8a65", "text": "RNG SWP",
            "tooltip": f"range sweep {sweep.level:.{digits}f}",
        })
    return out


def _interactive_markers(view: TFView, frame, setup, trend: int = 0) -> list[dict]:
    markers: list[dict] = []
    pattern_labels = {
        "hammer": "Hammer",
        "shooting_star": "Shooting star",
        "bull_engulf": "Bullish engulfing",
        "bear_engulf": "Bearish engulfing",
        "capitulation": "Capitulation",
    }
    for event in view.trigger_events():
        if not (frame.index[0] <= event.t <= frame.index[-1]):
            continue
        bull = event.direction > 0
        try:
            event_price = float(frame.loc[event.t, "high"] if bull
                                else frame.loc[event.t, "low"])
        except (KeyError, TypeError, ValueError):
            event_price = float(event.price)
        if event.kind == "BOS":
            text = (f"BOS{'↑' if bull else '↓'} {event.label}"
                    if event.label else "BOS")
            tooltip = (f"1m BOS close {'above' if bull else 'below'} "
                       f"{event.label} {event.level:.5f}")
        else:
            text = (f"CHoCH{'↑' if bull else '↓'} {event.label}"
                    if event.label else "CHoC")
            tooltip = (f"1m CHoCH close {event.price:.5f} through "
                       f"{event.level:.5f}")
        markers.append({
            "time": int(event.t.timestamp()),
            "price": event_price,
            "position": "aboveBar" if bull else "belowBar",
            "shape": "circle", "color": "#1b5e20" if bull else "#b71c1c",
            "text": text, "tooltip": tooltip,
        })
    for event in view.range_breakouts:
        if not (frame.index[0] <= event.t <= frame.index[-1]):
            continue
        bull = event.direction > 0
        markers.append({
            "time": int(event.t.timestamp()), "price": float(event.price),
            "position": "aboveBar" if bull else "belowBar",
            "shape": "circle", "color": "#ffd54f", "text": "RNG BRK",
            "tooltip": f"1m range breakout {'up' if bull else 'down'} {event.level:.5f}",
        })
    for sweep in view.sweeps:
        if sweep.source != "range" or not (frame.index[0] <= sweep.t <= frame.index[-1]):
            continue
        bull = sweep.direction > 0
        try:
            price = float(frame.loc[sweep.t, "close"])
        except (KeyError, TypeError, ValueError):
            price = float(sweep.level)
        markers.append({
            "time": int(sweep.t.timestamp()), "price": price,
            "position": "aboveBar" if bull else "belowBar",
            "shape": "circle", "color": "#ff8a65", "text": "RNG SWP",
            "tooltip": f"1m range sweep {sweep.level:.5f}",
        })
    for event in view.sweep_reversals:
        if not (frame.index[0] <= event.t <= frame.index[-1]):
            continue
        bull = event.direction > 0
        markers.append({
            "time": int(event.t.timestamp()), "price": float(event.price),
            "position": "aboveBar" if bull else "belowBar",
            "shape": "circle", "color": "#4dd0e1",
            "text": "HL SWP" if bull else "LH SWP",
            "tooltip": (f"1m {'HL' if bull else 'LH'} sweep reversal "
                        f"{event.swept:.5f}"),
        })

    atr_cut = atr(view.frame).reindex(frame.index)
    for pattern in detect_candle_patterns(frame, atr_cut):
        if pattern.idx >= len(frame):
            continue
        y = (float(frame["low"].iloc[pattern.idx]) if pattern.direction > 0
             else float(frame["high"].iloc[pattern.idx]))
        shape = "arrowUp" if pattern.direction > 0 else "arrowDown"
        color = "#ffd54f" if pattern.direction > 0 else "#ff8a65"
        if pattern.kind == "capitulation":
            shape, color = "circle", "#ffeb3b"
        elif pattern.kind.endswith("engulf"):
            shape = "circle"
            color = "#80cbc4" if pattern.direction > 0 else "#ef9a9a"
        markers.append({
            "time": int(frame.index[pattern.idx].timestamp()),
            "price": y,
            "position": "belowBar" if pattern.direction > 0 else "aboveBar",
            "shape": shape, "color": color, "text": "",
            "tooltip": pattern_labels.get(
                pattern.kind, pattern.kind.replace("_", " ").title()),
        })
    revs = detect_reversal_continuation(frame, atr_cut, trend)
    sweep_idxs = [sw.idx for sw in view.sweeps if sw.direction > 0]
    for idx in revs:
        if idx >= len(frame):
            continue
        after_sweep = any(0 <= idx - s_idx <= 3 for s_idx in sweep_idxs)
        markers.append({
            "time": int(frame.index[idx].timestamp()),
            "price": float(frame["low"].iloc[idx]),
            "position": "belowBar",
            "shape": "circle",
            "color": "#66bb6a" if after_sweep else "#a5d6a7",
            "text": "E/rev" if after_sweep else "rev",
            "tooltip": ("Reversal continuation after sweep" if after_sweep
                        else "Bullish reversal continuation"),
        })
    for pattern in view.hs[-2:]:
        times = [view.frame.index[i] for i in
                 (pattern.left_idx, pattern.head_idx, pattern.right_idx)]
        if any(not (frame.index[0] <= t <= frame.index[-1]) for t in times):
            continue
        try:
            hs_price = float(frame.loc[times[2], "high"] if pattern.direction < 0
                             else frame.loc[times[2], "low"])
        except (KeyError, TypeError, ValueError):
            hs_price = float(pattern.right_price)
        markers.append({
            "time": int(times[2].timestamp()),
            "price": hs_price,
            "position": "aboveBar" if pattern.direction < 0 else "belowBar",
            "shape": "square", "color": "#ce93d8",
            "text": "H&S done" if pattern.complete else "H&S forming",
            "tooltip": ("Head & shoulders complete" if pattern.complete
                        else "Head & shoulders forming"),
        })
    markers.sort(key=lambda item: item["time"])
    return markers


def _interactive_polylines(view: TFView, frame) -> list[dict]:
    lines: list[dict] = []
    for pattern in view.hs[-2:]:
        times = [view.frame.index[i] for i in
                 (pattern.left_idx, pattern.head_idx, pattern.right_idx)]
        if any(not (frame.index[0] <= t <= frame.index[-1]) for t in times):
            continue
        prices = [pattern.left_price, pattern.head_price, pattern.right_price]
        lines.append({
            "points": [{"time": int(t.timestamp()), "price": float(p)}
                       for t, p in zip(times, prices)],
            "color": "#ce93d8", "width": 1, "style": 0,
        })
        break_idx = (pattern.break_idx if pattern.break_idx is not None
                     else pattern.right_idx)
        neck_time = view.frame.index[break_idx]
        lines.append({
            "points": [{"time": int(times[0].timestamp()), "price": float(pattern.neckline)},
                       {"time": int(neck_time.timestamp()), "price": float(pattern.neckline)}],
            "color": "#ba68c8", "width": 1,
            "style": 0 if pattern.complete else 2,
        })
    return lines


def live_1m_candle(config: Config, symbol: str = "XAUUSD") -> dict | None:
    """Current forming 1m candle aggregated from the 1s MT4 feed."""
    path = Path(config.data_dir) / f"OHLCV_1s_{symbol}.csv"
    if not path.exists():
        return None
    try:
        feed, _ = parse_mt4_csv(path, config.server_utc_offset_hours,
                                config.auto_server_offset)
    except (OSError, ValueError):
        return None
    if len(feed) == 0:
        return None
    minute = feed.index[-1].floor("min")
    rows = feed[feed.index >= minute]
    if len(rows) == 0:
        return None
    return {
        "time": int(minute.timestamp()),
        "open": float(rows["open"].iloc[0]),
        "high": float(rows["high"].max()),
        "low": float(rows["low"].min()),
        "close": float(rows["close"].iloc[-1]),
    }


def _merge_live_candle(candles: list[dict], config: Config,
                       symbol: str = "XAUUSD") -> list[dict]:
    live = live_1m_candle(config, symbol)
    if live is None:
        return candles
    cutoff = live["time"]
    out = [c for c in candles if c["time"] < cutoff]
    out.append(live)
    return out


def build_interactive_payload(analyses: list[SymbolAnalysis],
                              config: Config) -> dict[str, dict]:
    payload: dict[str, dict] = {}
    for analysis in analyses:
        for tf in (240, 60, 5, 1):
            if tf not in analysis.views:
                continue
            view = analysis.views[tf]
            frame = _window(view, tf)
            price = float(frame["close"].iloc[-1])
            atr_v = float(view.atr.iloc[-1]) or 1.0
            htf = analysis.views.get(60)
            mtf = analysis.views.get(5)
            entry = analysis.views.get(1) or analysis.views.get(5)
            digits = 2 if price >= 10 else 5
            setup = analysis.setup

            candles = [
                {"time": int(ts.timestamp()), "open": float(o), "high": float(h),
                 "low": float(lo), "close": float(c)}
                for ts, o, h, lo, c in zip(frame.index, frame["open"], frame["high"],
                                           frame["low"], frame["close"])
            ]
            if tf == 1:
                candles = _merge_live_candle(candles, config, analysis.symbol)
            rsi = []
            for ts, value in view.rsi.reindex(frame.index).items():
                if value is not None and value == value and math.isfinite(float(value)):
                    rsi.append({"time": int(ts.timestamp()), "value": float(value)})

            price_lines: list[dict] = []
            price_lines += _swing_price_lines(view, frame, digits, "#ef5350",
                                              tf_label(tf))
            if htf is not None and tf != htf.tf:
                price_lines += _swing_price_lines(htf, frame, digits, "#ffb74d", "1h")
            if mtf is not None and tf in (1, 60):
                price_lines += _swing_price_lines(mtf, frame, digits, "#ffb74d", "5m")

            bands: list[dict] = _poi_bands(view, frame, view.atr_now,
                                           "#ef5350", tf_label(tf), limit=1)
            bands += _range_bands(view, frame, digits)
            markers: list[dict] = _swing_markers(view, frame, "#ef5350",
                                                 tf_label(tf), digits)
            if htf is not None and tf != htf.tf:
                markers += _swing_markers(htf, frame, "#ffb74d", "1h", digits,
                                          big=True)
            if mtf is not None and tf in (1, 60):
                markers += _swing_markers(mtf, frame, "#ffb74d", "5m", digits)
            if entry is None or tf != entry.tf:
                markers += _range_markers(view, frame, digits)
            polylines: list[dict] = []
            summary = ""
            if entry is not None and tf == entry.tf:
                if htf is not None:
                    bands += _poi_bands(htf, frame, htf.atr_now, "#ffb74d", "1h")
                if mtf is not None:
                    bands += _poi_bands(mtf, frame, mtf.atr_now, "#ffb74d", "5m")
                if setup is not None:
                    for value, color, label in ((setup.entry, "#42a5f5", "entry"),
                                                (setup.sl, "#ef5350", "SL"),
                                                (setup.tp, "#66bb6a", "TP")):
                        price_lines.append({
                            "price": float(value), "color": color, "width": 2,
                            "style": 0, "title": f"{label} {value:.{digits}f}",
                            "axis_label": True,
                        })
                markers += _interactive_markers(view, frame, setup,
                                                mtf.trend if mtf else 0)
                polylines = _interactive_polylines(view, frame)

            if setup is None and entry is not None and tf == entry.tf:
                atr_tf = float(view.atr.iloc[-1]) or 1.0
                pe = potential_entries(view, atr_tf)
                if pe.long_above is not None:
                    label = f"entry long {pe.long_above:.{digits}f}"
                    price_lines.append({
                        "price": float(pe.long_above), "color": "#66bb6a", "width": 1,
                        "style": 1, "title": label, "axis_label": True,
                    })
                if pe.short_below is not None:
                    label = f"entry short {pe.short_below:.{digits}f}"
                    price_lines.append({
                        "price": float(pe.short_below), "color": "#ef5350", "width": 1,
                        "style": 1, "title": label, "axis_label": True,
                    })
                if pe.retest is not None:
                    label = f"retest {pe.retest:.{digits}f}"
                    price_lines.append({
                        "price": float(pe.retest), "color": "#90caf9", "width": 1,
                        "style": 1, "title": label, "axis_label": True,
                    })
                    bands.append({
                        "y0": float(pe.retest - 0.08 * atr_tf),
                        "y1": float(pe.retest + 0.08 * atr_tf),
                        "from_time": int(frame.index[0].timestamp()),
                        "to_time": int(frame.index[-1].timestamp()),
                        "color": "#90caf9", "opacity": 0.13, "label": "",
                    })
                summary_bits = []
                if pe.long_above is not None:
                    summary_bits.append(f"long > {pe.long_above:.{digits}f}")
                if pe.short_below is not None:
                    summary_bits.append(f"short < {pe.short_below:.{digits}f}")
                if pe.retest is not None:
                    summary_bits.append(f"retest {pe.retest:.{digits}f}")
                summary = " · ".join(summary_bits)

            markers.sort(key=lambda marker: marker["time"])
            title = f"{analysis.symbol} {tf_label(tf)} — {analysis.bias_label}"
            title += (f" — {setup.direction_label} {setup.grade} ({setup.rr:.1f}R)"
                      if setup is not None else " — no trigger")
            legend_parts = [f"{tf_label(tf)} swings = red"]
            if htf is not None and tf != htf.tf:
                legend_parts.append("1h context = orange")
            if mtf is not None and tf in (1, 60):
                legend_parts.append("5m context = orange")
            legend_parts.append("POI = un-swept zone")
            if getattr(view, "ranges", None):
                legend_parts.append("range box = amber")
            if entry is not None and tf == entry.tf:
                legend_parts += ["BOS / CHoC / RNG BRK / RNG SWP / HL-LH SWP = 1m triggers",
                                 "H&S = 1m pattern",
                                 "rev = bullish reversal continuation",
                                 "E/rev = after a sweep"]
            legend = "  ·  ".join(legend_parts)
            payload[f"{analysis.symbol}|{tf_label(tf)}"] = {
                "symbol": analysis.symbol, "tf": tf_label(tf), "title": title,
                "digits": digits, "candles": candles, "rsi": rsi,
                "price_lines": price_lines, "bands": bands, "markers": markers,
                "polylines": polylines, "summary": summary, "legend": legend,
                "rsi_label": f"RSI 14 {view.rsi_now:.1f}",
                "rsi_floor": float(config.rsi_floor_for(analysis.symbol)),
                "rsi_ceiling": float(config.rsi_ceiling_for(analysis.symbol)),
            }
    return payload


def write_interactive_charts(analyses: list[SymbolAnalysis], config: Config,
                             out_dir: Path | None = None) -> Path:
    payload = build_interactive_payload(analyses, config)
    path = (out_dir or config.out_dir) / "interactive_charts.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, default=str), encoding="utf-8")
    return path
