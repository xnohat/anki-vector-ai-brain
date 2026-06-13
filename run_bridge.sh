#!/bin/bash
# Unified launcher: Vector Brain (our project) + wire-pod bridge.
#
# Flow when you say "Hey Vector, <Vietnamese>":
#   Vector mic -> wire-pod (VAD) -> our /stt (sensitive VN Whisper)
#             -> wire-pod intent match (native Vector Intents) OR our /v1 brain
#             -> Vector speaks Vietnamese (OpenAI voice) + animates (intents)
#
# Usage:  ./run_bridge.sh            (starts brain server + wire-pod bridge)
#         ./run_bridge.sh --no-pod   (start only the brain server)
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
WIREPOD="/home/pi/wire-pod/chipper"
BRAIN_PORT="${BRAIN_PORT:-7070}"

cd "$HERE"
[ -f .env ] && { set -a; . ./.env; set +a; }

echo "=== [0/3] Ensuring wire-pod has the Brain bridge ==="
# Applies our bundled bridge onto (upstream) wire-pod. No fork to maintain.
"$HERE/wirepod-bridge/install.sh" "$(dirname "$WIREPOD")"

echo "=== [1/3] Starting Vector Brain server (port $BRAIN_PORT) ==="
# Vietnamese STT model: 'small' is a good accuracy/speed balance on a Pi.
export VECTOR_STT_MODEL="${VECTOR_STT_MODEL:-small}"
"$HERE/.venv/bin/python" brain_server.py >/tmp/vector-brain.log 2>&1 &
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
sudo pkill -f "cmd/brain/main.go" 2>/dev/null || true
sudo pkill -f "cmd/vosk/main.go" 2>/dev/null || true
sudo pkill -f "chipper/start.sh" 2>/dev/null || true
sudo pkill -f "/tmp/go-build.*chipper" 2>/dev/null || true
sleep 2

echo "=== [3/3] Starting wire-pod Brain bridge (needs sudo for port 443) ==="
cd "$WIREPOD"
exec sudo -E ./start.sh
