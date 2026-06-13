# Implementation Plan — Smart Vietnamese Vector

**Status:** ✅ Implemented and verified on real hardware (2026-06-14).

This document records the architecture, the concrete changes made, and how each
piece was tested.

---

## 1. Architecture

```
                         ┌──────────────────────────────────────────────┐
                         │                  app.py                       │
                         │  (orchestrator: 2 threads + UI text input)    │
                         └──────────────────────────────────────────────┘
   ┌─────────────┐   wake word     ┌──────────────┐   intents     ┌──────────────┐
   │ USB mic ──▶ │ ───────────────▶│  CustomGPT    │──────────────▶│  Action       │
   │ Whisper STT │   (or UI text)  │  (gpt-5.5,    │  @CMD@ tokens │  Vector SDK   │
   │ speechstream│                 │   vision, VI) │               │  motors/anim  │
   └─────────────┘                 └──────┬───────┘               │  eyes/say     │
                                          │ @LOOK@                 └──────┬───────┘
   ┌─────────────┐  camera frame          ▼                              │ WAV
   │ Vector cam ─┼──────────────▶  multimodal vision               ┌─────▼───────┐
   │ Data.get_*  │                                                 │ Voice (TTS) │
   └─────────────┘                                                 │ OpenAI→16kHz│
                                                                   │ →robot spkr │
   Autonomous thread: every VECTOR_AGENT_INTERVAL s → gather_state │
   → CustomGPT.autonomous_tick() → optional @LOOK@ → act / @SILENT@└─────────────┘
```

### Two threads, one lock
- **conversation()** — consumes Whisper transcripts; engages the brain only on
  wake word or inside the active window.
- **agent_loop()** — autonomous ticks; skipped while a conversation is active.
- A single `Brain.lock` serializes all robot + brain access so the two threads
  never talk to Vector at once.

## 2. Files changed / added

| File | Change |
|------|--------|
| `src/customgpt.py` | **Rewritten.** Multimodal (`get_answer(query, image=)`), configurable `gpt-5.5`, `.env` loader, history cap, Vietnamese system prompt, `@LOOK@`/`@SILENT@` intents, `autonomous_tick()`. |
| `src/voice.py` | **New.** OpenAI TTS → resample 24k→16k → mono 16-bit WAV for Vector's speaker. |
| `src/whisperstt.py` | Vietnamese language (`VECTOR_STT_LANG=vi`), configurable model, proper `transcribe`/`translate` task. |
| `src/vectorbot.py` | `Action.say_wav()` streams a WAV to the speaker (waits on the future); safe `__del__`. |
| `src/ui.py` | Text box → brain (`on_submit` callback + Enter key) so it works without a mic. |
| `app.py` | **Rewritten.** Wake-word gating, active window, autonomous agent loop, LLM-triggered vision, Vietnamese `speak()` with mic-muting, optional mic, text input. |
| `.env` / `.env.example` | Key + all tunables (model, language, voice, intervals, wake words). |
| `run.sh` | Auto-loads `.env`. |
| `scripts/hw_smoketest.py` | **New.** Connect → see → speak Vietnamese hardware test. |

## 3. Intent / command vocabulary

The brain embeds tokens in its reply; `parse_commands()` extracts `@...@`.

| Token | Effect |
|-------|--------|
| `@FRWD_X@ @BACK_X@ @LEFT_X@ @RIGHT_X@ @STOP@` | Drive (X seconds) |
| `@LOOKUP_X@ @LOOKDOWN_X@ @CLAWUP_X@ @CLAWDOWN_X@` | Head / lift |
| `@EMOTE_HAPPY@ … @EMOTE_EYEROLL@` | Animation triggers |
| `@DETECT_START_X@ @DETECT_STOP_X@ @DETECT_STOP_ALL@` | YOLO detection (reserved) |
| `@LOOK@` | **LLM-triggered vision** — capture a frame, re-ask with the image |
| `@SILENT@` | Autonomous loop: stay quiet this tick |

`@LOOK@`/`@SILENT@` are control tokens and are filtered out before motor exec.

## 4. Cost / latency design
- Whisper STT runs **locally** (no API cost). Paid calls = wake-word turns +
  interval ticks only.
- Autonomous tick is **text-first**; an image is sent only if the LLM emits
  `@LOOK@`.
- Camera frames downscaled to ≤768px JPEG; stored frames replaced by a
  `[camera image]` placeholder so they are never resent.
- Conversation history capped at 24 messages.

## 5. Verification

### Unit / module (no robot)
- `py_compile` of all modules — OK.
- Wake-word detection on Vietnamese phrases — 5/5 correct.
- Vietnamese brain reply with intents — OK (`@EMOTE_SASSY@ @FRWD_1@ Tôi ổn…`).
- TTS WAV format — **mono / 16000 Hz / 16-bit** (Vector-compatible) — OK.

### Hardware (`scripts/hw_smoketest.py`, real Vector)
1. Connect via wire-pod — OK (battery 2).
2. Camera frame 640×360 — OK.
3. Vietnamese vision reply — *"Một người đàn ông ngồi trước bàn, trong phòng có
   màn hình và tủ."* (correctly described the real scene) — OK.
4. Vietnamese audio streamed to Vector's speaker — `playback ok: True` — OK.

## 6. How to run

```sh
cp .env.example .env   # set OPENAI_API_KEY (model/lang/voice already defaulted)
./run.sh               # auto-loads .env, launches the UI app
# Hardware smoke test:
.venv/bin/python scripts/hw_smoketest.py
```

## 7. Configuration knobs (`.env`)

`OPENAI_API_KEY`, `VECTOR_GPT_MODEL` (gpt-5.5), `VECTOR_LANG` (Vietnamese),
`VECTOR_STT_LANG`/`VECTOR_STT_MODEL`, `VECTOR_TTS`/`VECTOR_TTS_MODEL`/
`VECTOR_TTS_VOICE`, `VECTOR_WAKE_WORDS`, `VECTOR_AGENT_INTERVAL`,
`VECTOR_CONVO_WINDOW`, `VECTOR_AUTONOMOUS`.

## 7b. Microphone investigation (important finding)

Goal: let the user talk to Vector through **Vector's own microphone** (not a Pi mic).

- The SDK exposes a raw `AudioFeed` gRPC stream; `AudioFeedResponse.signal_power`
  is a `bytes` field documented as "Streamed audio data from the robot", which
  *should* be 16 kHz / 16-bit / mono (matches wire-pod's STT encoder).
- **However**, on this wire-pod firmware the feed is a **stub**: it emits a
  synthetic pure **1000 Hz sine test tone** (samples `[0,382,707,923,1000,…]`,
  FFT energy purity = 1.000), not the real mic. Confirmed by tone-playback
  calibration (gave contradictory rates) and a silent ambient capture.
- `src/vectormic.py` is implemented and ready, and **self-detects this stub**
  (spectral purity check) — it raises so `app.py` falls back to a USB mic / text.
  If a future firmware/SDK enables the real mic, it will start working with no
  code change.

**Working voice-input options today:**
1. **USB microphone on the Pi** — original design; `StreamHandler` + Whisper.
   (`libportaudio2` installed; plug in a mic and it works.)
2. **wire-pod voice pipeline** — wire-pod captures Vector's real mic for
   "Hey Vector"; integrate by forwarding wire-pod transcripts to this brain.
3. **Type in the UI** — works now, no mic needed.

## 8. Next steps / backlog
- Attach a USB mic to the Pi to enable the spoken wake-word path end to end
  (PortAudio installed; code already supports it).
- Wire `@DETECT_*@` to a live YOLO overlay in the UI camera panel.
- Explore Vector's onboard mic once the SDK fork implements `audio_feed`.
- Add simple task memory / multi-step intents (e.g. "đi tìm quả bóng").
