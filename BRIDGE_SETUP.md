# wire-pod Brain Bridge — Setup & Run

This wires **Vector's real microphone** (via wire-pod) to **our gpt-5.5 brain**,
using our own sensitive Vietnamese Whisper instead of wire-pod's STT.

## Architecture

```
"Hey Vector, <tiếng Việt>"
        │  (Vector's real mic + wake word, firmware)
        ▼
   wire-pod  ── VAD ──▶  POST /stt  ──▶  our Vietnamese Whisper   (brain_server.py)
        │                                   │ transcript
        │   ┌───────────────────────────────┘
        ▼   ▼
   intent match (native Vector Intents)  OR  POST /v1  ──▶ gpt-5.5 brain (Vietnamese,
        │ (forward/turn/fistbump/...)         │              emits {{intent}} tokens)
        ▼                                     ▼
   robot executes intent natively      wire-pod speaks Vietnamese (OpenAI "fable")
                                        + executes {{playAnimationWI||...}} etc.
```

- **wire-pod = the bridge**: Vector's mic, VAD, wake word, TTS, intent/animation execution.
- **our project = the brain**: STT (`/stt`) + LLM (`/v1`), both in `brain_server.py`.

## What was changed

| File | Change |
|------|--------|
| `brain_server.py` (our project) | `/stt` (VN Whisper) + `/v1/chat/completions` (gpt-5.5, streaming, intent tokens) |
| `src/customgpt.py` | `CustomGPT(system_prompt=...)` override for the bridge personality |
| `wirepod-bridge/stt_brain/Brain.go` | **bundled** STT service → posts Vector mic audio to `/stt` |
| `wirepod-bridge/cmd_brain/main.go` | **bundled** entrypoint for the bridge build |
| `wirepod-bridge/install.sh` | applies the bundle onto an upstream wire-pod (no fork) |
| `run.sh` (our project) | one-command unified launcher (installs bridge + runs) |

The bridge code lives **inside this project** (`wirepod-bridge/`). `install.sh`
copies it onto an upstream wire-pod checkout and edits that checkout's
`start.sh` (adds a `brain` branch) and `apiConfig.json` (knowledge → custom
endpoint `http://127.0.0.1:7070/v1`, STT → `brain`; non-secret fields only,
backup `apiConfig.json.bak-preBrain`). So there is **no wire-pod fork to
maintain** — pull upstream and re-run `install.sh`.

## Run it

```sh
cd ~/vector-advanced-ai
./run.sh
```

This (0) installs/updates the bridge into `~/wire-pod`, (1) starts the brain
server (Vietnamese Whisper `small` by default), (2) stops any running wire-pod,
and (3) launches the wire-pod brain bridge (needs sudo for port 443).

Then **say: "Hey Vector, … (tiếng Việt)"** and Vector replies in Vietnamese,
animating as he speaks.

### Useful env (in `.env`)
- `VECTOR_STT_MODEL` — Whisper size for `/stt` (`base` fast, `small`/`medium` more accurate).
- `VECTOR_GPT_MODEL` — brain model (default `gpt-5.5`).
- `BRAIN_PORT` — brain server port (default 7070).

## Revert to stock wire-pod
```sh
cd ~/wire-pod/chipper
sudo cp chipper.prebuilt.bak chipper
cp apiConfig.json.bak-preBrain apiConfig.json
# set STT_SERVICE back in source.sh, then sudo ./start.sh
```

## Notes / next steps
- Native intent **matching** currently runs en-US, so Vietnamese *movement*
  commands fall through to the brain (which still animates via `{{...}}`). For
  native Vietnamese movement intents, add `vi` intent-data to wire-pod.
- Camera vision in the voice path uses wire-pod's `{{getImage||front}}`; our
  standalone `app.py` path uses the SDK camera directly.
- Don't run `app.py` (SDK behavior control) and the wire-pod bridge against the
  robot at the same time without testing — two controllers.
