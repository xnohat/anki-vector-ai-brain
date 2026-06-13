# PRD — Smart Anki Vector with an LLM Brain

**Status:** Implemented & verified on hardware (Vector `Vector-S7H7` @ 192.168.1.105 via wire-pod)
**Owner:** xnohat
**Last updated:** 2026-06-14

---

## 1. Vision / Goal

Turn an Anki Vector robot into a genuinely *smart* companion robot whose "brain"
is a state-of-the-art OpenAI model. Vector should:

- **See** — understand what his camera is pointed at, in natural language.
- **Listen** — understand spoken commands (Vietnamese), only when addressed.
- **Talk** — reply out loud in natural Vietnamese through his own speaker.
- **Act** — move, emote, and react using the Vector SDK ("Vector intents").
- **Be autonomous** — on his own, periodically observe his environment and
  decide whether to react, without a human prompting him.

## 2. Background & Research

| Topic | Finding |
|-------|---------|
| SDK | Controlled via a wire-pod–compatible fork of the Vector Python SDK (`anki_vector`). Auth cert + GUID in `~/.anki_vector/`. |
| Server | Anki cloud is dead; the robot is driven by **wire-pod** (`~/wire-pod`) on the LAN. |
| Intents | Vector actions ("intents") = SDK calls: `behavior.say_text`, `motors.*`, `anim.play_animation_trigger`, `behavior.set_eye_color`, `camera`, `audio.stream_wav_file`. |
| Mic | The SDK's `enable_audio_feed` (Vector's own mic) is **stubbed** in this fork → must use a **USB mic on the Pi** + local Whisper, or type in the UI. |
| Speaker | `audio.stream_wav_file` works and accepts **8–16 kHz, 16-bit, mono WAV** → lets us play OpenAI-generated Vietnamese audio on the robot. |
| Onboard TTS | `say_text` is an English phoneme engine → Vietnamese sounds wrong → we replace it with OpenAI TTS. |
| Model | `gpt-5.5` confirmed available on the user's API key (vision-capable). |

## 3. Users & Use Cases

- **Primary user:** a Vietnamese-speaking owner who talks to Vector at home.
- *"Vector, bạn thấy gì?"* → Vector looks and describes the scene.
- *"Vector, tiến lên 2 giây"* → Vector drives forward.
- Idle: every ~90s Vector glances around and occasionally makes a remark.

## 4. Requirements

### Functional
1. **Wake-word gating** — Vector only sends audio to the brain when addressed by
   the wake word "vector" (incl. Vietnamese pronunciations: *véc tơ, vích to…*).
   A short active window lets follow-ups skip the wake word.
2. **Vietnamese I/O** — STT in Vietnamese (Whisper), brain replies in Vietnamese,
   TTS speaks Vietnamese on Vector's speaker.
3. **Conversational vision** — when asked to see/look/read, or when the LLM emits
   `@LOOK@`, capture a camera frame and let the model reason about it.
4. **Autonomous agent loop** — periodic ticks gather environment state
   (battery, charger, time, idle) and let the LLM decide: look, speak, emote, or
   stay silent (`@SILENT@`).
5. **Intents** — the model embeds `@COMMAND@` tokens (move/turn/lift/head/emote/
   detect/look/silent); the app parses and executes them via the SDK.
6. **No-mic fallback** — if no microphone, the user can type to Vector in the UI.

### Non-functional
- **Cost control:** local Whisper is free; paid GPT calls happen only on
  wake-word interactions and cheap interval ticks (text-first, vision only on
  demand). Camera frames are downscaled (≤768px) and not resent across turns.
- **Safety:** autonomous loop never interrupts an active conversation; thread
  access to the robot/brain is serialized by a lock.
- **Configurable:** everything via `.env` (model, language, voice, intervals).

## 5. Out of Scope (for now)
- Using Vector's onboard microphone (SDK stub — needs wire-pod-side work).
- SLAM / navigation, cube/accessory interaction, multi-turn task planning.
- Live YOLO bounding-box overlay (kept available but not the primary "vision").

## 6. Success Metrics
- ✅ Vector connects via wire-pod and answers in Vietnamese.
- ✅ Vector verbally describes what his camera sees (verified: *"Một người đàn
  ông ngồi trước bàn…"*).
- ✅ Vietnamese audio plays from Vector's speaker (verified on hardware).
- ✅ Brain only fires on wake word; autonomous loop ticks without a human.

## 7. Risks & Mitigations
| Risk | Mitigation |
|------|-----------|
| No USB mic attached | Optional mic; UI text input works today. |
| `gpt-5.5` disabled on an account | One-line `VECTOR_GPT_MODEL` override. |
| Mic hears Vector's own voice | Mic muted during playback. |
| API/network failure | Brain & TTS fail soft (fallback message / onboard voice). |
| Robot drives off a ledge | Movement intents are time-boxed; low-battery guard. |
