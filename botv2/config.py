"""Configuration for the MT4 -> strategy analysis pipeline."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# Default MT4 data folder (Windows). Overridable via MT4_DATA_DIR.
_HOME = Path.home()
MT4_DEFAULT_DIRS = [
    # macOS: MT4 under Wine (this is where MetaQuotes' Wine wrapper stores it)
    _HOME / "Library/Application Support/net.metaquotes.wine.metatrader4/drive_c/Program Files (x86)/MetaTrader 4/MQL4/Files",
    _HOME / "Library/Application Support/net.metaquotes.wine.metatrader5/drive_c/Program Files/MetaTrader 5/MQL5/Files",
    # Windows
    Path(r"C:\Program Files (x86)\MetaTrader 4\MQL4\Files"),
    Path(r"C:\Program Files\MetaTrader 4\MQL4\Files"),
    Path(r"C:\Program Files\MetaTrader 5\MQL5\Files"),
    Path(r"C:\Program Files (x86)\MetaTrader 5\MQL5\Files"),
]


def _env_path(name: str, default: Path) -> Path:
    raw = os.getenv(name)
    return Path(raw).expanduser() if raw else default


def default_data_dir() -> Path:
    """Resolve the data directory: env var, else first existing MT4 dir, else ./data."""
    env = os.getenv("MT4_DATA_DIR")
    if env:
        return Path(env).expanduser()
    for candidate in MT4_DEFAULT_DIRS:
        try:
            if candidate.is_dir():
                return candidate
        except OSError:
            continue
    return Path(__file__).resolve().parent.parent / "data"


@dataclass
class Config:
    """All tunables in one place.

    Defaults are derived from the user's trade journal (2026-03 .. 2026-10):
      * 5-min RSI floors observed between 12 and 32, ceilings between 62 and 85
      * 1-hour / 5-min swing levels used as liquidity magnets
      * 1-min bottoming/topping patterns (BOS / CHoCH / H&S / range sweep) as triggers
      * risk is expressed in R; stops are usually 1R, occasionally widened (and that
        habit shows up in the losing trades)
    """

    # --- data ---
    data_dir: Path = field(default_factory=default_data_dir)
    symbols: tuple[str, ...] = ("XAUUSD", "EURUSD", "BTCUSD")
    # Timeframe (minutes) -> filename suffix, e.g. XAUUSD_5.csv -> 5
    filename_pattern: str = "{symbol}_{timeframe}.csv"
    # Bars needed for a usable series
    min_bars: int = 60

    # --- timeframes ---
    # Higher timeframes give context, the entry timeframe gives the trigger.
    context_tfs: tuple[int, ...] = (240, 60)
    setup_tfs: tuple[int, ...] = (15, 5)
    entry_tf: int = 5
    auto_entry_tf: bool = True   # use 1m triggers when 1m data is exported
    # If the entry TF is 5m, 1m entry precision is unavailable; flagged in reports.

    # --- major swing filter (what gets marked on the charts) ---
    # atr_mult: prominence in ATRs; min_gap: bars between same-kind swings;
    # max_touches: wick touches allowed near the level (more = chop, not a swing).
    major_swing_params: dict = field(default_factory=lambda: {
        240: dict(atr_mult=1.0, min_gap=4, max_touches=3, tol_atr=0.35, window=60),
        60: dict(atr_mult=0.8, min_gap=4, max_touches=6, tol_atr=0.25, window=60),
        15: dict(atr_mult=1.3, min_gap=8, max_touches=3, tol_atr=0.35, window=100),
        5: dict(atr_mult=1.0, min_gap=6, max_touches=6, tol_atr=0.25, window=100),
        1: dict(atr_mult=1.8, min_gap=20, max_touches=3, tol_atr=0.30, window=200),
    })

    # --- real swing filter (1H and 5M) ---
    # Strict pivot (n bars each side, ties fail) + prominence on both sides +
    # isolation from nearby extremes. Overrides major_swing_params for these TFs.
    real_swing_params: dict = field(default_factory=lambda: {
        240: dict(n=4, prom=1.0, iso_band=0.35, prom_window=8, iso_window=20),
        60: dict(n=4, prom=1.0, iso_band=0.35, prom_window=8, iso_window=20),
        5: dict(n=5, prom=1.0, iso_band=0.35, prom_window=10, iso_window=25),
        1: dict(n=6, prom=1.0, iso_band=0.35, prom_window=12, iso_window=30),
    })

    # --- range detection (box detector from the reference script) ---
    range_tf: int = 5                 # detect ranges on the 5m frame
    range_min_bars: int = 24
    range_max_bars: int = 120
    range_k_height: float = 3.0       # height <= K * ATR
    range_er_max: float = 0.30        # Kaufman efficiency ratio ceiling
    range_tol_atr: float = 0.15       # boundary touch tolerance in ATR
    range_min_touch: int = 2          # touches at top and bottom
    range_break_atr: float = 0.25     # close beyond the box to end the range
    # Per-timeframe box windows (bars). Higher TFs need shorter windows in bars.
    range_params_by_tf: dict = field(default_factory=lambda: {
        1: dict(min_bars=24, max_bars=120, k_height=3.0, er_max=0.30,
                tol_atr=0.15, min_touch=2, break_atr=0.25),
        5: dict(min_bars=24, max_bars=120, k_height=3.0, er_max=0.30,
                tol_atr=0.15, min_touch=2, break_atr=0.25),
        60: dict(min_bars=12, max_bars=80, k_height=3.0, er_max=0.30,
                 tol_atr=0.15, min_touch=2, break_atr=0.25),
        240: dict(min_bars=12, max_bars=60, k_height=3.0, er_max=0.30,
                  tol_atr=0.15, min_touch=2, break_atr=0.25),
    })
    # Bars scanned back for box detection (keeps 1m detection fast).
    range_scan_bars: dict = field(default_factory=lambda: {
        1: 3000, 5: 5000, 60: 500, 240: 500,
    })
    range_sweep_tol_atr: float = 0.05 # min pierce beyond a box boundary
    range_reclaim_bars: int = 3       # bars allowed to close back inside
    range_break_close_atr: float = 0.25  # close beyond level to confirm breakout
    range_resolve_max_wait: int = 300 # bars to scan after a box

    # --- indicators ---
    rsi_period: int = 14
    atr_period: int = 14
    swing_k: int = 2                 # fractal strength (bars each side)
    divergence_min_bars: int = 5     # minimum spacing between swing points

    # --- strategy thresholds ---
    rsi_floor: float = 26.0          # 5-min floor (journal: 12..32)
    rsi_ceiling: float = 74.0        # 5-min ceiling (journal: 62..85)
    rsi_deep_floor: float = 18.0
    rsi_deep_ceiling: float = 82.0
    sweep_lookback: int = 20         # bars to look back for the swept level
    trigger_window: int = 12         # bars in which a trigger must have fired
    max_trigger_chase_atr: float = 1.0  # reject if price ran this far past the trigger

    # --- scoring (weights reflect the journal: divergence / swept 1h swing /
    #     1m H&S / range sweeps outperformed plain CHoC entries) ---
    min_score_watch: int = 3         # worth watching
    min_score_b: int = 4             # tradeable
    min_score_a: int = 7             # high conviction
    pts_rsi_extreme: int = 2
    pts_rsi_deep: int = 3
    pts_divergence: int = 3
    pts_hs: int = 3                  # 1-min head & shoulders neckline break (58% win)
    pts_bos: int = 2
    pts_choch: int = 2               # journal: plain CHoC entries underperformed
    pts_sweep: int = 2
    pts_range_sweep: int = 1         # extra when the swept level came from a range
    pts_htf_swing: int = 2
    pts_trend_alignment: int = 1
    pts_session: int = 1
    structural_trigger_required: bool = True
    require_rsi_extreme_or_divergence: bool = True
    # Journal: setups with >=2 pre-trade rules ticked win ~58% / PF 1.95,
    # one rule or none is a losing proposition. This gate is mandatory.
    min_top_rules: int = 2
    rsi_confirm_points: float = 3.0

    # --- journal calibration ---
    journal_db: Path | None = None          # auto-detected when None
    journal_calibration: bool = True        # use the DB's RSI zones / weights
    max_trades_per_day: int = 3             # account rule
    stop_after_consecutive_losses: int = 2  # -2R -> done for the day
    symbol_rsi_floor: dict = field(default_factory=dict)
    symbol_rsi_ceiling: dict = field(default_factory=dict)

    # --- risk ---
    max_sl_atr: float = 3.0          # stop must not exceed this many ATRs
    min_rr: float = 1.0              # reject setups below 1R
    target_rr: float = 2.0           # stretch target
    typical_risk_usd: float = 300.0  # fixed risk per trade for sizing and P&L

    # --- session / news ---
    # MT4 server time is usually GMT+2/+3; offset used to convert to UTC.
    server_utc_offset_hours: float = float(os.getenv("MT4_SERVER_UTC_OFFSET", "3"))
    auto_server_offset: bool = os.getenv("MT4_AUTO_OFFSET", "1") == "1"
    session_note: str = "sessions approximate (server offset configurable)"
    news_guard_minutes: int = 15
    news_file: Path | None = None    # optional CSV: datetime,currency,impact

    # --- cadence (seconds) ---
    # Report + charts + dashboard every 30 s. Change here or with TV_WATCH_INTERVAL_SEC.
    watch_interval_seconds: int = max(5, int(os.getenv("TV_WATCH_INTERVAL_SEC", "30")))
    charts_every_seconds: int = max(0, int(os.getenv("TV_CHARTS_EVERY_SEC", "30")))
    # once a trade is recommended/active, check it every 1 minute
    active_trade_interval_seconds: int = max(30, int(os.getenv("TV_ACTIVE_INTERVAL_SEC", "60")))
    # an active recommendation expires after this many hours even if neither level is hit
    active_plan_max_age_hours: float = float(os.getenv("TV_ACTIVE_PLAN_MAX_HOURS", "12"))
    # optional extra spacing between recommendations; 0 = only the open-trade gate
    recommendation_cooldown_minutes: float = float(os.getenv("TV_REC_COOLDOWN_MIN", "0"))
    # the dashboard page itself refreshes fast so it never lags behind a notification
    dashboard_refresh_seconds: int = max(5, int(os.getenv("TV_DASHBOARD_REFRESH_SEC", "30")))
    # v2 owns its own port so it can run alongside the original project (8765)
    dashboard_port: int = max(1, int(os.getenv("BOTV2_DASHBOARD_PORT", "8766")))

    # --- notifications ---
    notify_every_cycle: bool = os.getenv("TV_NOTIFY_EVERY_REPORT", "1") == "1"
    notify_sound: str = os.getenv("TV_NOTIFY_SOUND", "Glass")

    # --- freshness ---
    # MT4 writes a new 1m bar every minute; warn if the feed has stalled.
    stale_after_minutes: int = 6

    # --- output ---
    out_dir: Path = Path(__file__).resolve().parent.parent / "reports"
    charts_dir: Path | None = None   # defaults to out_dir/charts
    save_charts: bool = True
    chart_format: str = "png"        # png | html | both

    # --- LLM (optional) ---
    ai_base_url: str = os.getenv("AI_BASE_URL", "")
    ai_api_key: str = os.getenv("AI_API_KEY", "")
    ai_model: str = os.getenv("AI_MODEL", "")
    ai_timeout: int = 45

    # --- alerts (optional) ---
    telegram_token: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    telegram_chat: str = os.getenv("TELEGRAM_CHAT_ID", "")
    macos_notify: bool = os.getenv("MACOS_NOTIFY", "1") == "1"

    def rsi_floor_for(self, symbol: str) -> float:
        return float(self.symbol_rsi_floor.get(symbol.upper(), self.rsi_floor))

    def rsi_ceiling_for(self, symbol: str) -> float:
        return float(self.symbol_rsi_ceiling.get(symbol.upper(), self.rsi_ceiling))

    def symbol_dict(self) -> dict[str, tuple[str, ...]]:
        return {s: (s,) for s in self.symbols}

    @property
    def chart_dir(self) -> Path:
        return self.charts_dir or (self.out_dir / "charts")


DEFAULT = Config()
