"""Local dashboard server with the trade-result endpoint.

The dashboard is static HTML, but the tracking view needs to record a manual
HIT TP / HIT SL. This server serves the same files as ``python -m http.server``
and adds one endpoint:

    GET /api/mark?key=<plan-key>&status=hit_tp|hit_sl|recommended|open|closed|expired

It updates ``active_plans.json`` and rebuilds the dashboard, then redirects
back to the tracking view.
"""
from __future__ import annotations

import argparse
import json
import urllib.parse
from datetime import datetime, timezone
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ALLOWED_STATUSES = {"recommended", "open", "hit_tp", "hit_sl", "closed", "expired"}
CLOSED_STATUSES = {"hit_tp", "hit_sl", "closed", "expired"}


def mark_plan(directory: Path, key: str, status: str) -> bool:
    """Update one plan in ``directory/active_plans.json``."""
    path = directory / "active_plans.json"
    if not path.exists():
        return False
    try:
        plans = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    plan = plans.get(key) if isinstance(plans, dict) else None
    if not isinstance(plan, dict):
        return False
    from .report import set_plan_result

    set_plan_result(plan, status)
    if status in CLOSED_STATUSES:
        plan["closed_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    else:
        plan.pop("closed_at", None)
    path.write_text(json.dumps(plans, indent=2, default=str), encoding="utf-8")
    return True


def _rebuild(directory: Path) -> None:
    try:
        from .config import Config
        from .dashboard import build_dashboard

        config = Config()
        config.out_dir = directory
        build_dashboard(config)
    except Exception:
        pass


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, directory: str | None = None, **kwargs):
        super().__init__(*args, directory=directory, **kwargs)

    def end_headers(self) -> None:
        path = urllib.parse.urlparse(self.path).path.lower()
        if path.endswith((".html", ".json")):
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")
        super().end_headers()

    def do_GET(self) -> None:  # noqa: N802 - stdlib naming
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/interactive_1m_live.json":
            try:
                from .charts import live_1m_candle
                from .config import Config

                candle = live_1m_candle(Config())
                body = json.dumps({"key": "XAUUSD|1m", "candle": candle}).encode("utf-8")
            except Exception:
                body = b"{}"
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if parsed.path == "/api/mark":
            query = urllib.parse.parse_qs(parsed.query)
            key = query.get("key", [""])[0]
            status = query.get("status", [""])[0]
            if key and status in ALLOWED_STATUSES:
                directory = Path(self.directory)
                if mark_plan(directory, key, status):
                    try:
                        from .charts import regenerate_plan_charts
                        from .config import Config

                        plans = json.loads(
                            (directory / "active_plans.json").read_text(encoding="utf-8"))
                        plan = plans.get(key)
                        if isinstance(plan, dict):
                            config = Config()
                            config.out_dir = directory
                            plan["charts"] = regenerate_plan_charts(plan, config)
                            (directory / "active_plans.json").write_text(
                                json.dumps(plans, indent=2, default=str),
                                encoding="utf-8")
                    except Exception:
                        pass
                    _rebuild(directory)
            self.send_response(303)
            self.send_header("Location", "/dashboard.html")
            self.end_headers()
            return
        super().do_GET()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="botv2.serve")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--directory", required=True)
    args = parser.parse_args(argv)
    directory = str(Path(args.directory).expanduser())
    server = ThreadingHTTPServer(
        ("127.0.0.1", args.port),
        lambda *handler_args, **handler_kwargs: Handler(
            *handler_args, directory=directory, **handler_kwargs),
    )
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
