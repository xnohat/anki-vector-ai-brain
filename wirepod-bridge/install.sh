#!/bin/bash
# Install the Vector Brain bridge INTO a wire-pod checkout.
#
# We don't maintain a wire-pod fork: this script takes UPSTREAM wire-pod and
# drops in our small bridge (a custom STT service + cmd entry), then points its
# knowledge graph + STT at our brain server. Idempotent — safe to re-run.
#
# Usage:  ./install.sh [WIREPOD_DIR]      (default: ~/wire-pod)
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
WIREPOD_DIR="${1:-$HOME/wire-pod}"
UPSTREAM="https://github.com/kercre123/wire-pod"
CHIPPER="$WIREPOD_DIR/chipper"

echo "=== Vector Brain bridge installer ==="
echo "    wire-pod dir: $WIREPOD_DIR"

# 1. Get upstream wire-pod if it isn't there yet.
if [ ! -d "$CHIPPER" ]; then
    echo "    wire-pod not found -> cloning upstream ($UPSTREAM)"
    git clone "$UPSTREAM" "$WIREPOD_DIR"
fi

# 2. Drop in our bridge source (STT service + cmd entry).
mkdir -p "$CHIPPER/pkg/wirepod/stt/brain" "$CHIPPER/cmd/brain"
cp "$HERE/stt_brain/Brain.go"  "$CHIPPER/pkg/wirepod/stt/brain/Brain.go"
cp "$HERE/cmd_brain/main.go"   "$CHIPPER/cmd/brain/main.go"
echo "    installed brain STT service + cmd/brain"

# 3. Ensure start.sh has a 'brain' STT branch (idempotent).
if ! grep -q 'cmd/brain/main.go' "$CHIPPER/start.sh"; then
    python3 - "$CHIPPER/start.sh" <<'PY'
import sys
p = sys.argv[1]
s = open(p).read()
anchor = 'if [[ ${STT_SERVICE} == "leopard" ]]; then'
branch = (
    'if [[ ${STT_SERVICE} == "brain" ]]; then\n'
    '    if [[ -f ./chipper ]]; then\n'
    '        ./chipper\n'
    '    else\n'
    '        /usr/local/go/bin/go run -tags $GOTAGS -ldflags="${GOLDFLAGS}" cmd/brain/main.go\n'
    '    fi\n'
    '    elif [[ ${STT_SERVICE} == "leopard" ]]; then'
)
if anchor in s and 'cmd/brain/main.go' not in s:
    s = s.replace(anchor, branch, 1)
    open(p, 'w').write(s)
    print("    patched start.sh with brain branch")
PY
else
    echo "    start.sh already has brain branch"
fi

# 4. Point wire-pod at our brain (non-secret fields only; keeps existing keys).
CFG="$CHIPPER/apiConfig.json"
if [ -f "$CFG" ]; then
    [ -f "$CFG.bak-preBrain" ] || cp "$CFG" "$CFG.bak-preBrain"
    BRAIN_V1="${BRAIN_V1_URL:-http://127.0.0.1:7070/v1}"
    sed -i \
      -e 's#"provider":"openai"#"provider":"custom"#' \
      -e 's#"provider":"vosk"#"provider":"brain"#' \
      -e "s#\"endpoint\":\"\"#\"endpoint\":\"${BRAIN_V1//#/\\#}\"#" \
      -e 's#"id":"","model":""#"id":"","model":"gpt-5.5"#' \
      "$CFG"
    echo "    configured apiConfig.json -> knowledge=custom @ $BRAIN_V1, STT=brain"
else
    echo "    NOTE: $CFG not found yet. Set up wire-pod once (pair Vector, set your"
    echo "          OpenAI key in the wire-pod web UI), then re-run this installer."
fi

echo "=== done. Run the bridge with:  ../run_bridge.sh ==="
