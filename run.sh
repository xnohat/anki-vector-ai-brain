#!/bin/bash
# Vector Brain — start EVERYTHING with one command. Self-cleaning: it stops any
# running/zombie brain + wire-pod first (gracefully, so the SDK disconnects and
# the robot frees behaviour control) before starting fresh.
#
# Usage:  ./run.sh                    (brain server + wire-pod brain bridge)
#         ./run.sh --no-pod           (brain server only)
#         ./run.sh --install-daemon   (run 24/7 via systemd, start on boot)
#         ./run.sh --uninstall-daemon (remove the systemd service)
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
WIREPOD_DIR="${WIREPOD_DIR:-$HOME/wire-pod}"
CHIPPER="$WIREPOD_DIR/chipper"
BRAIN_PORT="${BRAIN_PORT:-7070}"

cd "$HERE"

# --------------------------------------------------------------------------- #
# Daemon setup: run the Vector Brain Agent 24/7 via systemd (starts on boot)
# --------------------------------------------------------------------------- #
SERVICE="vector-brain"
SERVICE_FILE="/etc/systemd/system/${SERVICE}.service"

if [ "$1" == "--install-daemon" ]; then
    echo "Installing ${SERVICE}.service (Vector Brain 24/7, starts on boot)..."
    sudo tee "$SERVICE_FILE" >/dev/null <<UNIT
[Unit]
Description=Vector Brain AI Agent (gpt-5.5 brain + wire-pod bridge)
Wants=network-online.target
After=network-online.target

[Service]
Type=simple
User=$(id -un)
Environment=HOME=$HOME
WorkingDirectory=$HERE
ExecStart=$HERE/run.sh
Restart=on-failure
RestartSec=10
TimeoutStopSec=25
KillSignal=SIGTERM

[Install]
WantedBy=multi-user.target
UNIT
    sudo systemctl daemon-reload
    sudo systemctl enable "$SERVICE"
    sudo systemctl restart "$SERVICE"
    echo "Done. The agent now runs 24/7 and on boot."
    echo "  status:  sudo systemctl status $SERVICE"
    echo "  stop:    sudo systemctl stop $SERVICE"
    echo "  logs:    journalctl -u $SERVICE -f   (brain: /tmp/vector-brain.log)"
    exit 0
fi

if [ "$1" == "--uninstall-daemon" ]; then
    echo "Removing ${SERVICE} daemon..."
    sudo systemctl disable --now "$SERVICE" 2>/dev/null || true
    sudo rm -f "$SERVICE_FILE"
    sudo systemctl daemon-reload
    echo "Removed. Use ./run.sh to start manually."
    exit 0
fi

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
# Let Vector's NATIVE firmware freeplay run when idle (the cute built-in
# behaviours). Stay-awake grabs control periodically and would suppress them, so
# it's OFF by default; set VECTOR_STAY_AWAKE=1 in .env only if reflexes pause.
export VECTOR_STAY_AWAKE="${VECTOR_STAY_AWAKE:-0}"
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

echo "=== [3/4] Starting wire-pod Brain bridge in the BACKGROUND (best-effort) ==="
# IMPORTANT: wire-pod is ONLY Vector's VOICE bridge (mic/STT/TTS on port 443). The
# BRAIN (port 7070 + the SDK body) is what actually controls Vector and MUST stay
# alive 24/7. So we no longer `exec` wire-pod as the service's main process — if
# wire-pod can't start (e.g. port 443 already taken by tailscale), it must NOT drag
# the brain down with it (that caused a restart crash-loop). The BRAIN is the
# resilient main process; wire-pod is best-effort and logged separately.
VECTOR_SERIAL="${VECTOR_SERIAL:-00907f6b}"
# Bind chipper's :443 to the Pi's CURRENT LAN IP so it coexists with tailscale
# Funnel holding <tailnet-ip>:443 (0.0.0.0:443 would collide). Vector reaches
# wire-pod via escapepod.local -> the Pi's LAN IP, so the bind must TRACK that IP.
lan_ip() { ip route get 1.1.1.1 2>/dev/null | grep -oP 'src \K\S+'; }

# wire-pod MANAGER (self-healing, background): keeps chipper running, bound to the
# Pi's current LAN IP, and AUTO RE-BINDS if that DHCP IP later changes (it does not
# hardcode an IP — it re-detects and restarts wire-pod on change). Also relaunches
# chipper if it dies. The BRAIN stays the resilient main process throughout.
: > /tmp/wire-pod.log
(
  cur=""
  while true; do
    now="$(lan_ip)"
    if { [ -n "$now" ] && [ "$now" != "$cur" ]; } || ! pgrep -x chipper >/dev/null 2>&1; then
      if [ -n "$cur" ] && [ -n "$now" ] && [ "$now" != "$cur" ]; then
        echo "    [lan] Pi LAN IP changed $cur -> $now; re-binding wire-pod to ${now}:443" \
          | tee -a /tmp/wire-pod.log
      fi
      sudo pkill -15 -x chipper 2>/dev/null || true
      sleep 2
      cur="$now"
      ( cd "$CHIPPER" && exec sudo -E WIREPOD_HOST="$cur" STT_SERVICE=brain \
          BRAIN_STT_URL="http://127.0.0.1:$BRAIN_PORT/stt" ./start.sh ) >>/tmp/wire-pod.log 2>&1 &
      echo "    wire-pod (re)started, bound ${cur:-0.0.0.0}:443 (log: /tmp/wire-pod.log)"
    fi
    sleep 30
  done
) &
WIREPOD_MGR_PID=$!
echo "    wire-pod manager pid $WIREPOD_MGR_PID (auto re-binds on LAN-IP change)"

# Re-apply the backpack-button -> "Hey Vector" voice-listening setting. A robot
# REBOOT resets button_wakeword to default, after which pressing the button no
# longer triggers STT. Re-apply it once wire-pod is up + connected to the robot.
(
  for i in $(seq 1 30); do
    if curl -sf --max-time 5 "http://localhost:8080/api-sdk/button_hey_vector?serial=$VECTOR_SERIAL" 2>/dev/null | grep -q done; then
      echo "    [button] backpack button -> Hey Vector voice listening: re-applied"
      break
    fi
    sleep 3
  done
) &

# Tie the SERVICE's life to the BRAIN only: clean up both children on stop; if the
# brain dies, exit (systemd Restart=on-failure brings it back); if wire-pod dies,
# the brain keeps running so Vector is never left without his controller.
cleanup() {
  kill "$WIREPOD_MGR_PID" 2>/dev/null || true   # stop the manager FIRST, else it re-launches chipper
  kill -15 "$BRAIN_PID" 2>/dev/null || true
  sudo pkill -15 -x chipper 2>/dev/null || true
  sudo pkill -15 -f "chipper/start.sh" 2>/dev/null || true
}
trap cleanup INT TERM EXIT

echo "=== [4/4] Brain is the resilient main process. Press the backpack button and speak. ==="
wait "$BRAIN_PID"; exit $?
