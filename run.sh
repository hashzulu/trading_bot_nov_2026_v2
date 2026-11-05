#!/bin/bash
# One-command entry point for trading_botv2.
#
#   ./run.sh                 analyse the live MT4 export
#   ./run.sh start           automation + dashboard + popup window
#   ./run.sh stop            stop everything
#   ./run.sh status          what is running / last report time
#   ./run.sh demo            analyse synthetic fixtures (no MT4 needed)
#
# Set MT4_DATA_DIR to override the export folder.
set -euo pipefail

cd "$(dirname "$0")"

if [ ! -d venv ]; then
    echo "creating virtualenv..."
    python3 -m venv venv
fi
# shellcheck disable=SC1091
source venv/bin/activate

if ! python -c "import pandas, markdown" >/dev/null 2>&1; then
    echo "installing dependencies..."
    pip install --quiet --upgrade pip
    pip install --quiet -r requirements.txt
fi

COMMAND="${1:-analyze}"
shift || true

case "$COMMAND" in
    start|up)
        echo "building reports, charts, dashboard and tracking..."
        python -u -m botv2 analyze --refresh-journal || true
        exec python -u -m botv2 start "$@"
        ;;
    stop|down)
        exec python -u -m botv2 stop "$@"
        ;;
    status)
        exec python -u -m botv2 status "$@"
        ;;
    demo)
        exec python -u -m botv2 demo --out data/demo --fresh "$@"
        ;;
    watch)
        if [ -n "${SECONDS:-}" ]; then
            exec python -u -m botv2 watch --seconds "$SECONDS" "$@"
        fi
        if [ -n "${INTERVAL:-}" ]; then
            exec python -u -m botv2 watch --interval "$INTERVAL" "$@"
        fi
        exec python -u -m botv2 watch "$@"
        ;;
    agent)
        exec python -u -m botv2 agent "$@"
        ;;
    doctor)
        exec python -u -m botv2 doctor "$@"
        ;;
    profile)
        exec python -u -m botv2 profile "$@"
        ;;
    recommendations|recs)
        exec python -u -m botv2 recommendations "$@"
        ;;
    dashboard|view)
        exec python -u -m botv2 dashboard "$@"
        ;;
    test)
        exec python -m pytest tests -q "$@"
        ;;
    analyze|*)
        exec python -u -m botv2 analyze "$@"
        ;;
esac
