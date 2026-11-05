"""Command line interface.

    python -m botv2 doctor                 # what data is present / missing
    python -m botv2 analyze                # read the charts, write reports
    python -m botv2 demo                   # synthetic MT4 fixtures + report
    python -m botv2 watch --interval 5     # poll and alert on new setups
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from .analysis import SymbolAnalysis, analyse_symbol
from .charts import render_all, write_interactive_charts
from .dashboard import build_dashboard, ensure_server, start_server, stop_server
from .config import Config, default_data_dir
from .data import discover, load_all, load_symbol, tf_label, write_demo_fixtures
from .journal import (
    apply_calibration,
    build_profile,
    find_database,
    load_profile,
    render_lessons_markdown,
)
from .report import (
    REC_LOG_NAME,
    append_recommendation,
    apply_active_plans,
    attach_trade_charts,
    console_summary,
    enforce_one_new_recommendation,
    ensure_recommendation_log,
    load_active_plans,
    recommendation_signature,
    record_active_plan,
    suppress_duplicate_setups,
    write_reports,
)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _config_from_args(args: argparse.Namespace) -> Config:
    config = Config()
    if getattr(args, "data_dir", None):
        config.data_dir = Path(args.data_dir).expanduser()
    if getattr(args, "symbols", None):
        config.symbols = tuple(s.strip().upper() for s in args.symbols.split(",") if s.strip())
    if getattr(args, "entry_tf", None):
        config.entry_tf = int(args.entry_tf)
    if getattr(args, "risk", None):
        config.typical_risk_usd = float(args.risk)
    if getattr(args, "out", None):
        config.out_dir = Path(args.out).expanduser()
    if getattr(args, "offset", None) is not None:
        config.server_utc_offset_hours = float(args.offset)
        config.auto_server_offset = False
    if getattr(args, "no_charts", False):
        config.save_charts = False
    return config


def _port(args: argparse.Namespace, config: Config) -> int:
    """Explicit --port wins, otherwise v2's dedicated default (8766)."""
    value = getattr(args, "port", None)
    return int(value) if value else int(config.dashboard_port)


def _file_url(path: Path) -> str:
    return path.resolve().as_uri()


def _link(path: Path, label: str) -> str:
    """Clickable terminal link (OSC 8) with the plain URL appended as fallback."""
    url = _file_url(path)
    return f"\033]8;;{url}\033\\{label}\033]8;;\033\\  {url}"


def _open_dashboard_url(url: str) -> None:
    """Open a URL as a chromeless popup window (Chrome app mode) if possible."""
    chrome = Path("/Applications/Google Chrome.app")
    try:
        if chrome.exists():
            flags = [
                f"--app={url}",
                # keep the popup refreshing even when it is not the focused window
                "--disable-background-timer-throttling",
                "--disable-renderer-backgrounding",
                "--disable-backgrounding-occluded-windows",
                "--disable-features=CalculateNativeWinOcclusion",
            ]
            subprocess.run(["open", "-na", "Google Chrome", "--args", *flags],
                           check=False, timeout=15)
            return
        subprocess.run(["open", url], check=False, timeout=15)
    except Exception:
        print(f"open this URL: {url}")


def _open_dashboard(path: Path) -> None:
    _open_dashboard_url(path.resolve().as_uri())


def _open_path(path: Path) -> None:
    """Best-effort open of the generated report (macOS `open`, else print path)."""
    try:
        if sys.platform == "darwin":
            subprocess.run(["open", str(path)], timeout=10, check=False)
        elif sys.platform.startswith("win"):
            os.startfile(str(path))  # type: ignore[attr-defined]
        else:
            subprocess.run(["xdg-open", str(path)], timeout=10, check=False)
    except Exception:
        print(f"open this file: {path}")


def _profile(config: Config, refresh: bool = False):
    """Load the journal profile (cached) and calibrate the strategy thresholds."""
    profile = load_profile(config, refresh=refresh)
    if profile is not None and config.journal_calibration:
        apply_calibration(config, profile)
    return profile


def _log_recommendations(analyses: list[SymbolAnalysis], config: Config,
                         profile=None) -> int:
    """Log and track every recommendation as one atomic step.

    The plan is recorded *before* the append-only log so a recommendation can
    never be announced without also appearing in Trade Tracking. Duplicate
    trigger/day keys are already tracked, so they are skipped.
    """
    ensure_recommendation_log(config)
    added = 0
    for analysis in analyses:
        setup = analysis.setup
        if setup is None or setup.locked_sig:
            continue
        if not record_active_plan(analysis, setup, config, profile=profile):
            continue
        if append_recommendation(analysis, setup, config):
            added += 1
        attach_trade_charts(analysis, setup, config)
    return added


def _analyse(config: Config) -> list[SymbolAnalysis]:
    data = load_all(config)
    analyses = [analyse_symbol(symbol_data, config) for symbol_data in data.values()]
    # pin live recommendations to the levels they were first logged with
    apply_active_plans(analyses, config)
    # a trigger/day that is already tracked is the same trade, not a new one
    suppress_duplicate_setups(analyses, config)
    # no second trade inside the cooldown window; suppressed ideas stay as watch items
    enforce_one_new_recommendation(analyses, config)
    analyses.sort(key=lambda a: (a.setup is None, a.symbol))
    return analyses


def _data_source(config: Config) -> str:
    found = discover(config.data_dir)
    if not found:
        return f"MT4 CSV ({config.data_dir}) — no files found"
    files = sum(len(tfs) for tfs in found.values())
    return f"MT4 CSV ({config.data_dir}) — {len(found)} symbol(s), {files} file(s)"


def _notify(title: str, message: str, config: Config, sound: str | None = None,
            open_url: str | None = None) -> None:
    print(f"\n🔔 {title}\n{message}")
    if open_url:
        print(f"   open: {open_url}")
    if not config.macos_notify:
        return
    # terminal-notifier gives clickable notifications (opens the report on click)
    local = (Path(__file__).resolve().parent.parent /
             "tools/terminal-notifier.app/Contents/MacOS/terminal-notifier")
    notifier = str(local) if local.exists() else shutil.which("terminal-notifier")
    if notifier:
        cmd = [notifier, "-title", title, "-message", message]
        if sound:
            cmd += ["-sound", sound]
        if open_url:
            cmd += ["-open", open_url]
        try:
            subprocess.run(cmd, timeout=8, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)
            return
        except Exception:
            pass
    try:
        extra = f' sound name {json.dumps(sound)}' if sound else ""
        script = (f'display notification {json.dumps(message)} '
                  f'with title {json.dumps(title)}{extra}')
        subprocess.run(["osascript", "-e", script], timeout=5,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass
    if config.telegram_token and config.telegram_chat:
        try:
            url = f"https://api.telegram.org/bot{config.telegram_token}/sendMessage"
            body = urllib.parse.urlencode({
                "chat_id": config.telegram_chat,
                "text": f"{title}\n{message}",
                "disable_web_page_preview": "true",
            }).encode()
            urllib.request.urlopen(url, data=body, timeout=8).read()
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #

def cmd_doctor(args: argparse.Namespace) -> int:
    config = _config_from_args(args)
    print(f"data dir : {config.data_dir}")
    print(f"exists   : {config.data_dir.exists()}")
    found = discover(config.data_dir)
    if not found:
        print("\nNo MT4 CSVs found. Expected names like:")
        for symbol in config.symbols:
            print(f"  {config.filename_pattern.format(symbol=symbol, timeframe=config.entry_tf)}")
        print("\nIn MT4: Tools → Options → Server, then Scripts/History export to MQL4/Files.")
        return 1

    for symbol in config.symbols:
        files = found.get(symbol)
        if not files:
            print(f"\n❌ {symbol}: no file  (expected "
                  f"{config.filename_pattern.format(symbol=symbol, timeframe=config.entry_tf)})")
            continue
        wanted = tuple(sorted({*config.context_tfs, *config.setup_tfs, config.entry_tf}))
        data = load_symbol(symbol, files, config, wanted)
        if data is None:
            print(f"\n❌ {symbol}: could not load")
            continue
        print(f"\n✅ {symbol}: base {tf_label(data.base_tf)} from {list(data.files.values())[0].name}")
        print(f"   coverage: {data.coverage()}")
        print(f"   frames  : {', '.join(tf_label(tf) + ('*' if tf in data.derived else '') for tf in data.available)}")
        for note in data.notes:
            print(f"   note    : {note}")
        if config.entry_tf not in data.frames:
            print(f"   ⚠️ entry TF {tf_label(config.entry_tf)} missing — triggers will use "
                  f"{tf_label(min(data.frames))}")
    print("\n(* = resampled from the finest file)")
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    config = _config_from_args(args)
    profile = _profile(config, refresh=getattr(args, "refresh_journal", False))
    analyses = _analyse(config)
    if not analyses:
        print("No data to analyse. Run `python -m botv2 doctor`.")
        return 1

    md_path, json_path = write_reports(analyses, config, _data_source(config),
                                       profile=profile)
    write_interactive_charts(analyses, config)
    charts = render_all(analyses, config)
    _log_recommendations(analyses, config, profile)
    build_dashboard(config)

    if getattr(args, "json", False):
        print(json_path.read_text(encoding="utf-8"))
    else:
        print(console_summary(analyses, config))
        stale = [a.symbol for a in analyses
                 if any("looks stale" in note for note in a.notes)]
        if stale:
            print(f"\n⚠ stale data for {', '.join(stale)} — open MT4, run ExportHistory, rerun")
        print(f"\nreport : {md_path}")
        print(f"data   : {json_path}")
        if charts:
            print("charts : " + ", ".join(str(p) for paths in charts.values()
                                         for p in paths))
        for analysis in analyses:
            for note in analysis.notes:
                print(f"note   : {analysis.symbol}: {note}")
        if profile is not None:
            print(f"journal: {profile.trades} trades, {profile.win_rate * 100:.0f}% win "
                  f"({Path(profile.source).name})")
    if getattr(args, "open", False):
        _open_path(md_path)
    return 0


def _report_blocks(markdown: str) -> list[tuple[str, list[str]]]:
    """Pull the SETUP/WATCHING/THESIS/NOTES blocks out of latest.md."""
    blocks: list[tuple[str, list[str]]] = []
    header, body, keep = None, [], False
    for line in markdown.splitlines():
        if line.startswith("### "):
            if header:
                blocks.append((header, [ln for ln in body if ln.strip()]))
            header, body, keep = line[4:].strip(), [], False
            continue
        if header is None:
            continue
        if line.startswith(("**SETUP**", "**WATCHING**", "**THESIS**", "**NOTES**",
                            "**COMPLIANCE**")):
            keep = True
            body.append(line)
            continue
        if keep:
            if line.startswith(("**Trade plan**", "**What is still missing**", "|", "**Levels")):
                keep = False
                continue
            body.append(line)
    if header:
        blocks.append((header, [ln for ln in body if ln.strip()]))
    return blocks


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    # a zombie still answers kill(0); treat it as dead
    try:
        stat = subprocess.run(["ps", "-p", str(pid), "-o", "stat="], capture_output=True,
                              text=True, timeout=3).stdout.strip()
        if stat.startswith("Z") or not stat:
            return False
    except Exception:
        pass
    return True


def _watch_pid_file(config: Config) -> Path:
    return config.out_dir / ".watch.pid"


def _running_watch_pid(config: Config) -> int | None:
    path = _watch_pid_file(config)
    if not path.exists():
        return None
    try:
        pid = int(path.read_text().split()[0])
    except (OSError, ValueError, IndexError):
        return None
    return pid if _pid_alive(pid) else None


def _spawn_watcher(config: Config, notify: bool = False):
    project = Path(__file__).resolve().parent.parent
    (project / "logs").mkdir(exist_ok=True)
    log = project / "logs" / "watch.log"
    cmd = [sys.executable, "-u", "-m", "botv2", "watch"]
    if not notify:
        cmd.append("--no-notify")
    with log.open("ab") as handle:
        proc = subprocess.Popen(cmd, cwd=str(project), stdout=handle, stderr=handle,
                                start_new_session=True)
    _watch_pid_file(config).write_text(f"{proc.pid}")
    return proc


def _supervisor_pid_file(config: Config) -> Path:
    return config.out_dir / ".supervisor.pid"


def cmd_supervise(args: argparse.Namespace) -> int:
    """Keep the watcher and the dashboard server alive."""
    config = _config_from_args(args)
    _supervisor_pid_file(config).write_text(str(os.getpid()))
    project = Path(__file__).resolve().parent.parent
    log = project / "logs" / "watch.log"
    watch_proc = None
    try:
        while True:
            if watch_proc is None or watch_proc.poll() is not None:
                if watch_proc is not None:
                    with log.open("ab") as handle:
                        handle.write(b"\n[supervisor] watcher exited - restarting\n")
                watch_proc = _spawn_watcher(config, notify=bool(getattr(args, "notify", False)))
                with log.open("ab") as handle:
                    handle.write(f"[supervisor] watcher pid {watch_proc.pid}\n".encode())
            try:
                ensure_server(config.out_dir, _port(args, config))
            except Exception:
                pass
            time.sleep(10)
    except KeyboardInterrupt:
        return 0
    finally:
        if watch_proc is not None and watch_proc.poll() is None:
            watch_proc.terminate()
        _supervisor_pid_file(config).unlink(missing_ok=True)


def cmd_start(args: argparse.Namespace) -> int:
    """Start the watcher (detached) + dashboard server + popup window."""
    config = _config_from_args(args)
    project = Path(__file__).resolve().parent.parent
    (project / "logs").mkdir(exist_ok=True)

    sup_path = _supervisor_pid_file(config)
    sup_pid = None
    if sup_path.exists():
        try:
            candidate = int(sup_path.read_text().strip())
            if _pid_alive(candidate):
                sup_pid = candidate
        except (OSError, ValueError):
            pass
    if sup_pid:
        print(f"supervisor already running (pid {sup_pid})")
    else:
        log = project / "logs" / "watch.log"
        with log.open("ab") as handle:
            sup = subprocess.Popen(
                [sys.executable, "-u", "-m", "botv2", "supervise",
                 "--port", str(_port(args, config))]
                + (["--notify"] if getattr(args, "notify", False) else []),
                cwd=str(project), stdout=handle, stderr=handle, start_new_session=True)
        sup_path.write_text(str(sup.pid))
        time.sleep(2.0)
        if _pid_alive(sup.pid):
            print(f"supervisor started (pid {sup.pid}) — restarts the watcher if it dies")
        else:
            print("supervisor failed to start — check logs/watch.log")
            return 1

    pid = _running_watch_pid(config)
    if pid:
        print(f"automation: running (pid {pid}) — logs/watch.log")
    else:
        time.sleep(3)
        pid = _running_watch_pid(config)
        print(f"automation: {'running (pid %d)' % pid if pid else 'starting…'} — logs/watch.log")

    ensure_recommendation_log(config)
    path = build_dashboard(config)
    if path is None:
        path = config.out_dir / "dashboard.html"
    try:
        base_url, port = start_server(config.out_dir, _port(args, config))
        url = f"{base_url}/dashboard.html"
        print(f"dashboard: {url}")
        if not getattr(args, "no_open", False):
            _open_dashboard_url(url)
    except Exception as exc:
        print(f"dashboard server failed ({exc}) — build only")
        if not getattr(args, "no_open", False):
            _open_dashboard(path)
    print("stop everything with: ./run.sh stop")
    return 0


def cmd_stop(args: argparse.Namespace) -> int:
    """Kill the watcher and the dashboard server."""
    config = _config_from_args(args)
    stopped = []
    sup_path = _supervisor_pid_file(config)
    if sup_path.exists():
        try:
            sup_pid = int(sup_path.read_text().strip())
            if _pid_alive(sup_pid):
                os.kill(sup_pid, 15)
                stopped.append(f"supervisor (pid {sup_pid})")
        except (OSError, ValueError):
            pass
        sup_path.unlink(missing_ok=True)
    time.sleep(0.5)
    pid = _running_watch_pid(config)
    if pid:
        try:
            os.kill(pid, 15)
            stopped.append(f"automation (pid {pid})")
        except OSError:
            pass
    _watch_pid_file(config).unlink(missing_ok=True)
    if subprocess.run(["pkill", "-f", "botv2 watch"], check=False,
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
        stopped.append("stray watcher(s)")
    if stop_server(config.out_dir):
        stopped.append("dashboard server")
    print("stopped: " + ", ".join(stopped) if stopped else "nothing was running")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    config = _config_from_args(args)
    sup_path = _supervisor_pid_file(config)
    sup_pid = None
    if sup_path.exists():
        try:
            candidate = int(sup_path.read_text().strip())
            sup_pid = candidate if _pid_alive(candidate) else None
        except (OSError, ValueError):
            sup_pid = None
    print(f"supervisor: {'running (pid %d)' % sup_pid if sup_pid else 'stopped'}")
    pid = _running_watch_pid(config)
    print(f"automation: {'running (pid %d)' % pid if pid else 'stopped'}")
    pid_path = config.out_dir / ".dashboard_server.pid"
    if pid_path.exists():
        try:
            port, spid = pid_path.read_text().split()[:2]
            print(f"dashboard:  http://127.0.0.1:{port}/dashboard.html (pid {spid})")
        except (OSError, ValueError):
            print("dashboard:  unknown state")
    else:
        print("dashboard:  stopped")
    report = config.out_dir / "latest.md"
    if report.exists():
        print(f"last report: {datetime.fromtimestamp(report.stat().st_mtime):%H:%M:%S}")
    return 0


def cmd_recommendations(args: argparse.Namespace) -> int:
    config = _config_from_args(args)
    path = Path(config.out_dir) / REC_LOG_NAME
    if not path.exists():
        print(f"no recommendations yet ({path})")
        return 0
    text = path.read_text(encoding="utf-8")
    if getattr(args, "tail", 0):
        blocks = [b.strip() for b in text.split("\n---\n") if b.strip()]
        header = blocks[0] if blocks and blocks[0].startswith("# Recommended") else ""
        body = blocks[1:] if header else blocks
        text = (header + "\n\n---\n\n" if header else "") + "\n\n---\n\n".join(
            body[-int(args.tail):]) + "\n"
    print(text)
    return 0


def cmd_dashboard(args: argparse.Namespace) -> int:
    config = _config_from_args(args)
    out_dir = config.out_dir
    if getattr(args, "stop", False):
        stopped = stop_server(out_dir)
        print("dashboard server stopped" if stopped else "no dashboard server running")
        return 0

    path = build_dashboard(config)
    if path is None:
        print("No report yet — run `./run.sh analyze` first.")
        return 1
    print(f"dashboard: {path}")

    if getattr(args, "file", False):
        # direct file:// works only if the browser has Desktop-folder permission
        if not getattr(args, "no_open", False):
            _open_dashboard(path)
        else:
            print(f"open: {path.resolve().as_uri()}")
        return 0

    port = _port(args, config)
    try:
        base_url, port = start_server(out_dir, port)
    except Exception as exc:
        print(f"could not start server ({exc}); falling back to file://")
        if not getattr(args, "no_open", False):
            _open_dashboard(path)
        return 0
    url = f"{base_url}/dashboard.html"
    print(f"serving: {url}")
    if not getattr(args, "no_open", False):
        _open_dashboard_url(url)
    return 0


def cmd_chat(args: argparse.Namespace) -> int:
    """Chat-ready update: the journal blocks + absolute links to charts/report."""
    config = _config_from_args(args)
    md_path = config.out_dir / "latest.md"
    if not md_path.exists():
        print("No report yet — run `./run.sh analyze` first.")
        return 1
    markdown = md_path.read_text(encoding="utf-8")
    charts_dir = config.chart_dir.resolve()
    report = md_path.resolve()

    stamp = ""
    for line in markdown.splitlines():
        if line.startswith("**Generated:**"):
            stamp = line.replace("**Generated:**", "").strip().rstrip("  ")
            break
    lines = [f"**2-min update — {stamp}**", ""]

    blocks = [b for b in _report_blocks(markdown) if any(
        ln.startswith("**SETUP**") for ln in b[1])]
    for header, body in blocks:
        symbol = header.split(" ")[0]
        lines.append(f"**{header}**")
        lines.append("")
        for line in body:
            if line.startswith("**") and lines and lines[-1] != "":
                lines.append("")
            lines.append(line)
            if line.startswith("**") and not line.startswith(("**SETUP**", "**WATCHING**",
                                                             "**THESIS**", "**NOTES**")):
                lines.append("")
        links = " · ".join(
            f"[{tf}]({(charts_dir / f'{symbol}_{tf}.png').as_posix()})"
            for tf in ("1h", "5m", "1m")
            if (charts_dir / f"{symbol}_{tf}.png").exists()
        )
        lines.append(f"charts: {links} · [full report]({report.as_posix()})")
        lines.append("")

    if not blocks:
        lines.append(markdown.strip())
    print("\n".join(lines).rstrip())
    return 0


def cmd_profile(args: argparse.Namespace) -> int:
    config = _config_from_args(args)
    db = find_database(config)
    if db is None:
        print("Journal DB not found. Set JOURNAL_DB=/path/to/TradingJournalDB.sqlite")
        return 1
    try:
        profile = build_profile(db)
    except (ValueError, OSError, sqlite3.Error) as exc:
        print(f"Could not read the journal: {exc}")
        return 1
    config.out_dir.mkdir(parents=True, exist_ok=True)
    (config.out_dir / "journal_profile.json").write_text(
        json.dumps(profile.to_dict(), indent=2), encoding="utf-8")
    markdown = render_lessons_markdown(profile)
    lessons_path = config.out_dir / "journal_lessons.md"
    lessons_path.write_text(markdown, encoding="utf-8")
    (config.out_dir / "journal_profile.json").write_text(
        json.dumps(profile.to_dict(), indent=2), encoding="utf-8")
    print(markdown)
    print(f"saved: {lessons_path}")
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    config = _config_from_args(args)
    demo_dir = Path(args.out).expanduser() if args.out else config.data_dir
    if args.out:
        config.out_dir = demo_dir / "reports"
    config.data_dir = demo_dir
    if args.fresh or not any(demo_dir.glob("*.csv")):
        written = write_demo_fixtures(demo_dir, config, bars=args.bars)
        print("wrote demo fixtures:")
        for path in written:
            print(f"  {path}")
    analyses = _analyse(config)
    md_path, json_path = write_reports(analyses, config, "synthetic MT4-format demo data",
                                       profile=None)
    write_interactive_charts(analyses, config)
    charts = render_all(analyses, config)
    _log_recommendations(analyses, config)
    build_dashboard(config)
    print(console_summary(analyses, config))
    print(f"\nreport : {md_path}")
    print(f"data   : {json_path}")
    if charts:
        print("charts : " + ", ".join(str(p) for p in charts.values()))
    if getattr(args, "open", False):
        _open_path(md_path)
    return 0


def _csv_mtimes(config: Config) -> dict[str, float]:
    found = discover(config.data_dir)
    stamps: dict[str, float] = {}
    for tfs in found.values():
        for path in tfs.values():
            try:
                stamps[str(path)] = path.stat().st_mtime
            except OSError:
                pass
    return stamps


def _status_line(analyses: list[SymbolAnalysis], note: str = "") -> str:
    stamp = datetime.now().strftime("%H:%M:%S")
    parts = []
    for analysis in analyses:
        if analysis.setup:
            parts.append(f"{analysis.symbol} {analysis.setup.direction_label} "
                         f"{analysis.setup.grade} {analysis.setup.rr:.1f}R")
        else:
            parts.append(f"{analysis.symbol} {analysis.bias_label.split(' (')[0]}")
    return f"[{stamp}] " + " · ".join(parts) + (f" · {note}" if note else "")


def cmd_watch(args: argparse.Namespace) -> int:
    config = _config_from_args(args)
    profile = _profile(config)
    seconds_arg = getattr(args, "seconds", None)
    if seconds_arg is not None:
        interval_seconds = max(5, int(seconds_arg))
    elif args.interval is not None:                       # legacy minutes flag
        interval_seconds = max(5, int(args.interval) * 60)
    else:
        interval_seconds = config.watch_interval_seconds
    chart_arg = getattr(args, "charts_every", None)
    chart_every_seconds = max(0, int(chart_arg * 60) if chart_arg is not None
                              else config.charts_every_seconds)
    quiet = bool(getattr(args, "quiet", False))
    skip_unchanged = bool(getattr(args, "skip_unchanged", False))

    state_path = config.out_dir / "alerted_setups.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    if state_path.exists():
        try:
            seen = set(json.loads(state_path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            seen = set()

    last_mtimes: dict[str, float] = {}
    last_chart_ts = 0.0
    if not quiet:
        cadence = f"{interval_seconds}s" if interval_seconds < 60 else f"{interval_seconds // 60} min"
        chart_text = (f"every {chart_every_seconds}s" if 0 < chart_every_seconds < 60
                      else f"every {chart_every_seconds // 60} min" if chart_every_seconds
                      else "on new setups only")
        print(f"watching {config.data_dir} every {cadence} — reports in {config.out_dir}")
        print(f"ctrl-c to stop. charts: on new setup or {chart_text}.")

    while True:
        mtimes = _csv_mtimes(config)
        changed = mtimes != last_mtimes
        if (not changed) and skip_unchanged and not getattr(args, "once", False):
            if not quiet:
                print(_status_line([], "no new bars (waiting)"))
            time.sleep(interval_seconds)
            continue

        cycle_started = time.time()
        analyses = _analyse(config)
        if not analyses:
            if not quiet:
                print("no data — check MT4 export")
            if getattr(args, "once", False):
                return 1
            time.sleep(interval_seconds)
            continue

        write_reports(analyses, config, _data_source(config), profile=profile)
        write_interactive_charts(analyses, config)
        last_mtimes = mtimes

        # Deduplicate alerts by the stable trigger/day key, not the drifting
        # entry/grade signature: the same trade must not re-alert with new levels.
        # A locked plan has already been announced, and a trigger/day that is
        # already in active_plans is already tracked, so neither alerts again.
        tracked_sigs = set(load_active_plans(config))
        new_setups = [(a, a.setup) for a in analyses
                      if a.setup and not a.setup.locked_sig
                      and recommendation_signature(a, a.setup) not in seen
                      and recommendation_signature(a, a.setup) not in tracked_sigs]
        now = time.time()
        # a 2-minute cadence means every cycle gets fresh charts
        want_charts = bool(new_setups) or (
            chart_every_seconds and (chart_every_seconds <= interval_seconds
                                     or (now - last_chart_ts) >= chart_every_seconds))
        if want_charts:
            render_all(analyses, config)
            last_chart_ts = now
        _log_recommendations(analyses, config, profile)
        # dashboard HTML is cheap: rebuild every cycle so it never lags a notification
        build_dashboard(config)

        stale = [a.symbol for a in analyses
                 if any("looks stale" in n for n in a.notes)]
        note = "STALE DATA" if stale else (
            f"{len(new_setups)} new setup(s)" if new_setups else
            ("no new bars" if not changed else "updated"))
        if not quiet:
            print(_status_line(analyses, note))
            if not getattr(args, "no_links", False):
                report_path = config.out_dir / "latest.md"
                print("  report: " + _link(report_path, "latest.md"))
                for analysis in analyses:
                    parts = []
                    for tf_label_ in ("1h", "5m", "1m"):
                        chart = config.chart_dir / f"{analysis.symbol}_{tf_label_}.png"
                        if chart.exists():
                            parts.append(_link(chart, tf_label_))
                    if parts:
                        print(f"  {analysis.symbol} charts: " + " | ".join(parts))

        if config.notify_every_cycle and not getattr(args, "no_notify", False):
            if new_setups:
                pass  # the setup alert below is the notification for this cycle
            else:
                plain = " · ".join(
                    (f"{a.symbol} {a.setup.direction_label} {a.setup.grade}"
                     if a.setup else f"{a.symbol} {a.bias_label.split(' (')[0]}")
                    for a in analyses
                )
                stamp = datetime.now().strftime("%H:%M")
                _notify(f"Report ready — {stamp}", plain, config,
                        sound=config.notify_sound,
                        open_url=_file_url(config.out_dir / "latest.md"))

        for analysis, setup in new_setups:
            seen.add(recommendation_signature(analysis, setup))
            if getattr(args, "no_notify", False):
                continue
            message = (
                f"{setup.symbol} {setup.direction_label} {setup.grade} @ {setup.entry:.5f} "
                f"| SL {setup.sl:.5f} | TP {setup.tp:.5f} | {setup.rr:.1f}R\n"
                f"top rules {setup.rule_count}/5 · {setup.entry_note}"
            )
            _notify(f"Setup: {setup.symbol} {setup.direction_label}", message, config,
                    sound=config.notify_sound,
                    open_url=_file_url(config.chart_dir / f"{setup.symbol}_{tf_label(setup.trigger_tf)}.png")
                    if (config.chart_dir / f"{setup.symbol}_{tf_label(setup.trigger_tf)}.png").exists()
                    else _file_url(config.out_dir / "latest.md"))

        state_path.write_text(json.dumps(sorted(seen), indent=2), encoding="utf-8")

        if getattr(args, "once", False):
            return 0
        active_setup = any(a.setup is not None for a in analyses)
        if active_setup:
            wait = max(interval_seconds, config.active_trade_interval_seconds)
        else:
            wait = interval_seconds
        if not quiet and active_setup:
            print(f"  trade active — next update in {wait // 60} min")
        elapsed = time.time() - cycle_started
        time.sleep(max(3.0, wait - elapsed))


def cmd_agent(args: argparse.Namespace) -> int:
    """Install/uninstall a launchd login agent that keeps reports current."""
    config = _config_from_args(args)
    project = Path(__file__).resolve().parent.parent
    run_script = project / "run.sh"
    logs = project / "logs"
    logs.mkdir(exist_ok=True)

    label = "com.botv2.watch"
    if getattr(args, "action", "install") == "uninstall":
        target = Path.home() / "Library/LaunchAgents" / f"{label}.plist"
        subprocess.run(["launchctl", "unload", str(target)], check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if target.exists():
            target.unlink()
        print(f"removed {target}" if not target.exists() else "could not remove the agent")
        return 0

    seconds = max(5, int(args.interval) * 60 if args.interval is not None
                  else config.watch_interval_seconds)
    plist = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>{label}</string>
  <key>ProgramArguments</key>
  <array>
    <string>/bin/bash</string>
    <string>{run_script}</string>
    <string>watch</string>
    <string>--seconds</string><string>{seconds}</string>
    <string>--quiet</string>
  </array>
  <key>WorkingDirectory</key><string>{project}</string>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>StandardOutPath</key><string>{logs / 'watch.log'}</string>
  <key>StandardErrorPath</key><string>{logs / 'watch.err.log'}</string>
  <key>EnvironmentVariables</key>
  <dict><key>PATH</key><string>/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin</string></dict>
</dict>
</plist>
"""
    target = Path.home() / "Library/LaunchAgents" / f"{label}.plist"
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        target.write_text(plist, encoding="utf-8")
        subprocess.run(["launchctl", "load", "-w", str(target)], check=False)
        print(f"installed {target}")
        print(f"reports refresh every {seconds}s; log: {logs / 'watch.log'}")
        print("uninstall: ./run.sh agent --uninstall")
    except OSError as exc:
        fallback = project / f"{label}.plist"
        fallback.write_text(plist, encoding="utf-8")
        print(f"could not write to LaunchAgents ({exc}).")
        print(f"saved to {fallback}; install with:")
        print(f"  cp '{fallback}' ~/Library/LaunchAgents/ && launchctl load -w "
              f"~/Library/LaunchAgents/{label}.plist")
    return 0


# --------------------------------------------------------------------------- #
# parser
# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="botv2", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--data-dir", help="folder with MT4 CSVs "
                       "(default: env MT4_DATA_DIR or the MT4 install)")
        p.add_argument("--symbols", help="comma separated, e.g. XAUUSD,EURUSD,BTCUSD")
        p.add_argument("--entry-tf", type=int, help="entry timeframe in minutes (default 5)")
        p.add_argument("--risk", type=float, help="nominal risk per trade in USD")
        p.add_argument("--offset", type=float, help="MT4 server UTC offset in hours")
        p.add_argument("--out", help="report output folder")
        p.add_argument("--no-charts", action="store_true", help="skip chart images")

    doctor = sub.add_parser("doctor", help="inspect available data")
    common(doctor)
    doctor.set_defaults(func=cmd_doctor)

    analyze = sub.add_parser("analyze", help="analyse charts and write the report")
    common(analyze)
    analyze.add_argument("--json", action="store_true", help="print JSON instead of summary")
    analyze.add_argument("--refresh-journal", action="store_true",
                         help="rebuild the journal profile from the database")
    analyze.add_argument("--open", action="store_true",
                         help="open the generated report when done (macOS)")
    analyze.set_defaults(func=cmd_analyze)

    start = sub.add_parser("start", help="run automation + dashboard with one command")
    common(start)
    start.add_argument("--no-open", action="store_true", help="do not open the popup window")
    start.add_argument("--notify", action="store_true", help="enable desktop notifications")
    start.add_argument("--port", type=int, default=None,
                       help="local server port (default: 8766)")
    start.set_defaults(func=cmd_start)

    sup = sub.add_parser("supervise", help=argparse.SUPPRESS)
    common(sup)
    sup.add_argument("--port", type=int, default=None)
    sup.add_argument("--notify", action="store_true", help=argparse.SUPPRESS)
    sup.set_defaults(func=cmd_supervise)

    stop = sub.add_parser("stop", help="stop automation + dashboard")
    common(stop)
    stop.set_defaults(func=cmd_stop)

    status = sub.add_parser("status", help="show what is running")
    common(status)
    status.set_defaults(func=cmd_status)

    recs = sub.add_parser("recommendations", help="show the append-only recommendation log")
    common(recs)
    recs.add_argument("--tail", type=int, default=0, help="only the last N entries")
    recs.set_defaults(func=cmd_recommendations)

    dashboard = sub.add_parser("dashboard", help="build and open the one-page dashboard")
    common(dashboard)
    dashboard.add_argument("--no-open", action="store_true", help="build only, do not open a window")
    dashboard.add_argument("--file", action="store_true",
                           help="open via file:// instead of the local server")
    dashboard.add_argument("--port", type=int, default=None,
                           help="local server port (default: 8766)")
    dashboard.add_argument("--stop", action="store_true", help="stop the local dashboard server")
    dashboard.set_defaults(func=cmd_dashboard)

    chat = sub.add_parser("chat", help="print the latest report summary with chart links")
    common(chat)
    chat.set_defaults(func=cmd_chat)

    profile_cmd = sub.add_parser("profile", help="mine the journal DB for edges and leaks")
    common(profile_cmd)
    profile_cmd.set_defaults(func=cmd_profile)

    demo = sub.add_parser("demo", help="run on synthetic MT4 fixtures")
    common(demo)
    demo.add_argument("--bars", type=int, default=3200)
    demo.add_argument("--fresh", action="store_true", help="overwrite existing fixtures")
    demo.add_argument("--open", action="store_true", help="open the report when done")
    demo.set_defaults(func=cmd_demo)

    watch = sub.add_parser("watch", help="keep reports fresh and alert on new setups")
    common(watch)
    watch.add_argument("--seconds", type=int, default=None,
                       help="seconds between reports (default: config, currently 30)")
    watch.add_argument("--interval", type=int, default=None,
                       help="minutes between reports (legacy; overrides --seconds)")
    watch.add_argument("--charts-every", type=int, default=None,
                       help="re-render charts every N minutes (default: config)")
    watch.add_argument("--skip-unchanged", action="store_true",
                       help="only report when the CSV actually changed")
    watch.add_argument("--quiet", action="store_true", help="no status lines")
    watch.add_argument("--no-notify", action="store_true",
                       help="disable desktop notifications")
    watch.add_argument("--no-links", action="store_true",
                       help="do not print clickable report/chart links each cycle")
    watch.add_argument("--once", action="store_true",
                       help="run one cycle then exit (for cron/launchd)")
    watch.set_defaults(func=cmd_watch)

    agent = sub.add_parser("agent", help="install/uninstall a login agent for continuous reports")
    common(agent)
    agent.add_argument("action", nargs="?", default="install",
                       choices=["install", "uninstall"])
    agent.add_argument("--interval", type=int, default=None,
                       help="minutes between reports (default: config)")
    agent.set_defaults(func=cmd_agent)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
