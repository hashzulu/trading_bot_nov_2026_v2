# trading_botv2

A focused port of the **Report + Charts** and **Trade Tracking** workflows from
`tradingview-automation`. It reads the same MT4 CSV exports, mines the same
`TradingJournalDB.sqlite`, runs the same structure/confluence engine, and produces
recommended trades with frozen entry / SL / TP levels.

The dashboard has exactly two views:

| View | What it does |
|---|---|
| **Report + Charts** | The full markdown report beside annotated 1m/5m/1h/4h PNGs and an interactive Lightweight Charts panel (candles, RSI, levels, POIs, structure, plan levels). |
| **Trade Tracking** | One row per recommendation with frozen levels, status, CET open/close times, USD P&L at a fixed risk, R multiple, and a click-through report snapshot. **HIT TP / HIT SL** buttons persist the result. |

## Quick start

```bash
./run.sh start      # build once, then automation + dashboard + popup
./run.sh stop       # stop automation and the dashboard server
./run.sh status     # what is running / last report time
./run.sh            # analyse once (writes reports/ + dashboard.html)
./run.sh demo       # run on synthetic fixtures (no MT4 needed)
./run.sh doctor     # which CSVs were found / missing
./run.sh test       # test suite
```

`start` launches a supervisor that restarts the watcher and dashboard server if
either dies. Reports refresh on the configured cadence (30s by default; 60s while
a trade is live).

### Runs independently of v1

v2 uses its own port (**8766**), its own `reports/`, `logs/`, pid files and
process names (`botv2`), so it can run alongside the original project on 8765
without stopping or interfering with it. Override the port with
`--port` or `BOTV2_DASHBOARD_PORT`.

## Data

Same sources as the original project:

- **MT4 exports** — auto-detected from the MT4/MQL4 `Files` folder, or set
  `MT4_DATA_DIR=/path/to/Files`. Export `1,5,15,60,240` for best precision.
  Only `_1` is required; coarser frames are resampled.
- **Journal** — `TradingJournalDB.sqlite` is auto-detected in the parent
  `Trading-Journal` workspace (or set `JOURNAL_DB=...`). It calibrates RSI zones,
  edges/leaks and the ≥2-top-rules gate.

## Swing detection (4H / 1H / 5M / 1M)

`botv2.features.real_swings` is the detector used for 4H, 1H, 5M and 1M levels, charts and
swing-based rules. A swing must pass three filters: a strict pivot (higher/lower than
`n` bars each side, ties fail), prominence of at least `1.0` ATR on both sides, and
isolation (`0.35` ATR) from nearby equal extremes. Defaults are `n=4` on 4H and 1H, `n=5` on
5M and `n=6` on 1M; tune them in `Config.real_swing_params`.

Each Report + Charts PNG draws its own timeframe's real swings in red; 1H context
levels are orange and 5M context levels are blue.

The **Demo** tab shows 1H / 5M / 1M swing-detection screenshots generated from the live
`XAUUSD_1.csv` export (falling back to `data/demo/` when no live file is present). Only
un-swept levels are drawn: once price trades through a swing, it is no longer a live
swing point and both its marker and line are removed. Range-detection cards draw the
5M box plus the resolved breakout or sweep; if recent bars have no range, the image
goes back to the most recent one.

The 1M BOS used by the strategy is the same one drawn in the images. It only fires
at a trend extreme: **bearish context** = a final LL plus an LH, then a 1m close
above the most recent **un-swept** LH; **bullish context** = a final HH plus an HL,
then a 1m close below the most recent **un-swept** HL. A level already traded
through by an earlier candle is stale and skipped. Once the BOS has printed, later
highs/lows are continuation, not another BOS. Real 1H/5M/1M swings feed chart
levels, POIs, potential entries, level clustering, and the swing-based rule checks;
the 1M BOS feeds the entry trigger and the interactive-chart markers.

1M CHoCH also fires only at a trend extreme: **bearish context** = final LL → high
→ higher low → close above that high; **bullish context** = final HH → low → lower
high → close below that low. Later highs/lows are continuation, not another CHoCH.
CHoCH events are shown as purple diamonds in the Demo image and as `CHoCH↑` /
`CHoCH↓` markers on the 1M PNG and interactive chart.

5-min RSI confirmation is not standalone: the prior swing must carry a same-direction
RSI divergence. A price HL with an RSI HL only ticks the confirmation rule if the
preceding swing already formed the divergence (for example price LL / RSI HL).

## 1-minute primary triggers

- **H&S** neckline break
- **BOS** against the most recent un-swept LH/HL
- **CHoCH** at a trend extreme
- **Range breakout** — a 1m close beyond a tight prior range, continuing
- **Range sweep** — a 1m break of the range boundary that closes back inside

Each timeframe detects its own boxes: **1M 24–120 bars**, **5M 24–120 bars**,
**1H 12–80 bars**, **4H 12–60 bars**; height ≤ 3 ATR, efficiency ratio ≤ 0.30,
at least 2 touches at each boundary, 0.15 ATR touch tolerance, and a 0.25 ATR
close beyond the box to end it.

After a box is found, a forward resolver uses the frozen high/low: a pierce of
0.05 ATR opens a 3-bar reclaim window; a close back inside is a **range sweep**,
otherwise a close 0.25 ATR beyond the boundary is a **range breakout**. The
resolver scans up to 300 bars and ignores ambiguous two-sided pierces.
- **HL/LH sweep** — sweep of the HL/LH followed by a close back through the
  prior high/low

When 1M data is unavailable, 5M becomes the fallback entry timeframe and uses
the same extreme/un-swept BOS and CHoCH logic on the 5M frame. H&S uses the same
frame-agnostic neckline detector.

## Live 1M candle

The interactive 1M chart keeps completed candles from `XAUUSD_1.csv`. The current
forming minute is aggregated from `OHLCV_1s_XAUUSD.csv` (open/high/low/close/tick
volume), replacing any partial 1M candle for that minute. The dashboard polls
`/interactive_1m_live.json` every second and updates only that last candle.

## How a recommendation becomes a tracked trade

1. `botv2.analysis` finds a setup (bias + structure break + momentum/liquidity
   confluence + journal rule gate) and builds the plan.
2. `botv2.report.record_active_plan` writes it to `reports/active_plans.json`
   **before** it is announced, with the report snapshot and chart paths.
3. `append_recommendation` adds an append-only block to
   `reports/recommended_trades.md`.
4. Later cycles call `apply_active_plans`, which pins the original levels and
   auto-detects `hit_tp` / `hit_sl` / `expired`.
5. The dashboard's Trade Tracking view reads `active_plans.json`; the local
   server exposes `/api/mark?key=...&status=hit_tp|hit_sl` for manual results.

Only one trade is open at a time. A new recommendation is made only after the
live plan hits TP/SL or expires (`active_plan_max_age_hours`, default 12h).
Re-detections of the same trigger/day are silent, so levels never drift.

## Output

Written to `reports/`:

- `latest.md` / `latest.json` — report (markdown + structured)
- `interactive_charts.json` — candles, RSI, price lines, markers and overlays
- `charts/<SYMBOL>_<tf>.png` — annotated PNGs per timeframe
- `active_plans.json` — live and closed recommendations, frozen levels, snapshots
- `recommended_trades.md` — append-only recommendation log
- `journal_lessons.md` / `journal_profile.json` — journal mining output
- `dashboard.html` — the generated two-tab UI

## Layout

```
botv2/
  config.py      tunables + MT4 folder discovery
  data.py        MT4 CSV parser, timeframe discovery/resampling, demo fixtures
  features.py    RSI/ATR, fractals, real swings (4H/1H/5M/1M), BOS/CHoCH, H&S, sweeps, divergence, sessions
  demo.py        Demo-tab swing screenshots from the live XAUUSD 1m export
  journal.py     mines the journal DB: RSI zones, edges/leaks, rule matrix, gate
  analysis.py    bias, levels, confluence scoring, rule gate, plan, checklist
  narrative.py   thesis/notes (+ optional LLM polish)
  report.py      markdown/JSON, recommendation log, active-plan lifecycle
  charts.py      annotated PNGs + interactive chart payload
  dashboard.py   two-tab dashboard (Report + Charts, Trade Tracking)
  serve.py       local static server + manual result endpoint
  cli.py         doctor / analyze / demo / watch / start / stop / status
tests/           parsing, resampling, structure, patterns, rule gate, dashboard
mt4/             ExportHistory.mq4
```

## Notes

- No execution: it produces a plan, placing orders stays manual.
- News is a manual pre-trade check; there is no economic-calendar feed.
- If only 5m data is exported, triggers fall back to 5m and the report says so.
- Optional LLM polish: `AI_BASE_URL`, `AI_API_KEY`, `AI_MODEL`.
- Desktop notifications: `TV_NOTIFY_EVERY_REPORT`, `TV_NOTIFY_SOUND`, Telegram env vars.
