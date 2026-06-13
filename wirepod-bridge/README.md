# wirepod-bridge

The Vector Brain bridge for wire-pod — bundled here so there is **no wire-pod
fork to maintain**. `install.sh` drops these files onto an *upstream* wire-pod
checkout and points it at our brain server.

## Files
- `stt_brain/Brain.go` — a wire-pod STT "service" named `brain`. Instead of
  transcribing on-device, it sends Vector's real microphone audio
  (`req.DecodedMicData`, 16 kHz mono) to our `POST /stt` (sensitive Vietnamese
  Whisper in `brain_server.py`). Replaces wire-pod's weak built-in STT.
- `cmd_brain/main.go` — the build entrypoint that registers the `brain` service.
- `install.sh` — copies the two files into a wire-pod checkout
  (`pkg/wirepod/stt/brain/`, `cmd/brain/`), adds a `brain` branch to
  `start.sh`, and edits `apiConfig.json` (non-secret fields only) so wire-pod's
  knowledge graph calls our `POST /v1` brain. Clones upstream wire-pod if absent.

## Install
```sh
./install.sh [WIREPOD_DIR]     # default ~/wire-pod
```
Then run everything from the project root with `../run.sh`.

## Why a bridge?
On this firmware the SDK's `AudioFeed` is a stub (a 1 kHz test tone), so the only
way to Vector's real microphone is wire-pod. We let wire-pod capture the mic +
do VAD + speak (OpenAI voice) + execute Vector Intents, while **our project is
the brain** (STT + gpt-5.5). See `../BRIDGE_SETUP.md` for the full picture.

## Updating wire-pod
Pull upstream in your wire-pod checkout, then re-run `install.sh` (idempotent):
```sh
cd ~/wire-pod && git pull && cd - && ./install.sh
```
