#!/bin/bash
# Vector Brain — start EVERYTHING with one command.
#
# Flow when you say "Hey Vector, <Vietnamese>":
#   Vector mic -> wire-pod (VAD) -> our /stt (sensitive VN Whisper)
#             -> wire-pod intent match (native Vector Intents) OR our /v1 brain
#             -> Vector speaks Vietnamese (OpenAI voice) + animates (intents)
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

echo "=== [0/3] Ensuring wire-pod has the Brain bridge ==="
# Applies our bundled bridge onto (upstream) wire-pod. No fork to maintain.
"$HERE/wirepod-bridge/install.sh" "$WIREPOD_DIR"

echo "=== [1/3] Starting Vector Brain server (port $BRAIN_PORT) ==="
# Fast cloud STT (OpenAI) so the robot's voice request doesn't time out -> the
# Pi's local Whisper was too slow and caused the "network disconnected" face.
export VECTOR_STT_BACKEND="${VECTOR_STT_BACKEND:-openai}"
"$HERE/.venv/bin/python" -u brain_server.py >/tmp/vector-brain.log 2>&1 &
BRAIN_PID=$!
echo "    brain server pid $BRAIN_PID (log: /tmp/vector-brain.log)"

echo "    waiting for brain to be ready..."
until curl -s --max-time 3 "http://127.0.0.1:$BRAIN_PORT/health" >/dev/null 2>&1; do
    if ! kill -0 "$BRAIN_PID" 2>/dev/null; then
        echo "    ERROR: brain server died. See /tmp/vector-brain.log"; exit 1
    fi
    sleep 2
done
echo "    brain server is up."

if [ "$1" == "--no-pod" ]; then
    echo "=== brain only (--no-pod). Ctrl+C to stop. ==="
    wait "$BRAIN_PID"; exit 0
fi

echo "=== [2/3] Stopping any running wire-pod ==="
sudo pkill -x chipper 2>/dev/null || true
sudo pkill -f "cmd/brain/main.go" 2>/dev/null || true
sudo pkill -f "cmd/vosk/main.go" 2>/dev/null || true
sudo pkill -f "chipper/start.sh" 2>/dev/null || true
sudo pkill -f "/tmp/go-build.*chipper" 2>/dev/null || true
sleep 2

echo "=== [3/3] Starting wire-pod Brain bridge (needs sudo for port 443) ==="
cd "$CHIPPER"
exec sudo -E STT_SERVICE=brain BRAIN_STT_URL="http://127.0.0.1:$BRAIN_PORT/stt" ./start.sh
