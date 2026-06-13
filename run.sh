#!/bin/bash
# Vector Brain — start EVERYTHING with one command. Self-cleaning: it stops any
# running/zombie brain + wire-pod first (gracefully, so the SDK disconnects and
# the robot frees behaviour control) before starting fresh.
#
# Usage:  ./run.sh             (brain server + wire-pod brain bridge)
#         ./run.sh --no-pod    (brain server only)
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
WIREPOD_DIR="${WIREPOD_DIR:-$HOME/wire-pod}"
CHIPPER="$WIREPOD_DIR/chipper"
BRAIN_PORT="${BRAIN_PORT:-7070}"

cd "$HERE"
[ -d .venv ] || { echo "No .venv — run: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"; exit 1; }
[ -f .env ] && { set -a; . ./.env; set +a; }

# --------------------------------------------------------------------------- #
# [0] Stop everything cleanly (kill zombies, free the robot's SDK control)
# --------------------------------------------------------------------------- #
echo "=== [0/4] Stopping any running/zombie brain + wire-pod ==="

# Brain: SIGTERM first so SmartVector disconnects the SDK cleanly (a SIGKILL
# leaves a stale control connection that blocks the next start for ~60s).
if pgrep -f "brain_server.py" >/dev/null; then
    echo "    stopping brain (graceful)..."
    pkill -15 -f "brain_server.py" 2>/dev/null || true
    for i in $(seq 1 12); do
        pgrep -f "brain_server.py" >/dev/null || break
        sleep 1
    done
    pkill -9 -f "brain_server.py" 2>/dev/null || true   # force any straggler
fi

# wire-pod (runs as root).
echo "    stopping wire-pod..."
sudo pkill -15 -x chipper 2>/dev/null || true
sudo pkill -15 -f "chipper/start.sh" 2>/dev/null || true
sleep 2
sudo pkill -9 -x chipper 2>/dev/null || true
sudo pkill -9 -f "chipper/start.sh" 2>/dev/null || true

# Wait for the ports to actually free.
for i in $(seq 1 10); do
    ss -tlnp 2>/dev/null | grep -qE ":$BRAIN_PORT|:443" || break
    sleep 1
done
echo "    give the robot a moment to release behaviour control..."
sleep 5
echo "    clean."

echo "=== [1/4] Ensuring wire-pod has the Brain bridge ==="
"$HERE/wirepod-bridge/install.sh" "$WIREPOD_DIR"

echo "=== [2/4] Starting Vector Brain server (port $BRAIN_PORT) ==="
# Fast cloud STT (OpenAI) so the robot's voice request doesn't time out.
export VECTOR_STT_BACKEND="${VECTOR_STT_BACKEND:-openai}"
"$HERE/.venv/bin/python" -u brain_server.py >/tmp/vector-brain.log 2>&1 &
BRAIN_PID=$!
echo "    brain server pid $BRAIN_PID (log: /tmp/vector-brain.log)"
echo "    waiting for brain to connect to Vector + be ready..."
until curl -s --max-time 3 "http://127.0.0.1:$BRAIN_PORT/health" >/dev/null 2>&1; do
    if ! kill -0 "$BRAIN_PID" 2>/dev/null; then
        echo "    ERROR: brain server died. See /tmp/vector-brain.log"; exit 1
    fi
    sleep 2
done
echo "    brain server is up."

if [ "$1" == "--no-pod" ]; then
    echo "=== brain only (--no-pod). Ctrl+C to stop. ==="
    trap 'kill -15 "$BRAIN_PID" 2>/dev/null' INT TERM
    wait "$BRAIN_PID"; exit 0
fi

echo "=== [3/4] Starting wire-pod Brain bridge (needs sudo for port 443) ==="
cd "$CHIPPER"
echo "=== [4/4] Say 'Hey Vector' / press the backpack button and speak. ==="
exec sudo -E STT_SERVICE=brain BRAIN_STT_URL="http://127.0.0.1:$BRAIN_PORT/stt" ./start.sh
