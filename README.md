# vector-advanced-ai

> A smart Anki Vector **robot pet** with a gpt-5.5 brain — it listens with its own
> mic, talks and thinks in Vietnamese, sees, feels touch, senses its world, and
> acts with its body. One brain, one body, one process.

![vector](resources/vector.jpg)

## What it does
- **Listens** through Vector's *real* microphone ("Hey Vector, …") via wire-pod.
- **Thinks** with one gpt-5.5 brain (Vietnamese, sassy-but-loving pet personality).
- **Acts** with its body: come near you (`@APPROACH@` — turns to your face, rolls
  closer), cuddle like a happy dog (hand + wheels), raise its hand, look around,
  drive/turn, change eye colour, emote.
- **Feels** the backpack touch and being picked up → cuddles and reacts.
- **Senses** everything (battery, proximity, cliff, faces, held/picked-up, …) and
  an **autonomous agent loop** lets the LLM judge it and do delightful unexpected
  things — roll over and say *"Em nhớ anh quá"*, raise a hand, look around.
- **Sees** through its camera when you ask.

## How it works

```
 Mic ──▶ Whisper STT ──▶ ┌──────────────────────────────┐ ──▶ TTS (Vector voice)
                         │   GPT brain (gpt-5.5, vision)  │
 Camera frame ─────────▶ │   src/customgpt.py            │ ──▶ @COMMAND@ intents
                         └──────────────────────────────┘      (move / emote / detect)
```

- **Listen** — `src/speechstream.py` records mic audio and transcribes it locally with Whisper.
- **Think** — `src/customgpt.py` sends the transcript (and, when asked, a live camera frame) to a vision-capable OpenAI model.
- **See** — when you say things like *"what do you see"*, *"look"*, *"read this"*, *"what colour is this"*, the app grabs a frame from Vector's camera and the model describes/reads/reasons about it in real time.
- **Act** — the model embeds `@COMMAND@` intent tokens in its reply; `app.py` strips them out and drives Vector (movement, lift, head, emotes, object detection) via the SDK.

### Behaviour

- **Wake word** — Vector ignores you until you say **"Vector"** (Vietnamese pronunciations like *véc tơ / vích to* are matched too). Whisper runs locally so listening is free; only wake-word turns hit the paid API. After replying he stays active for a few seconds so follow-ups need no wake word.
- **Autonomous agent loop** — every ~90s Vector senses his situation (battery, charger, time, idle) and the LLM decides what to do: glance around with his camera (`@LOOK@`), make a short remark, emote, or stay quiet (`@SILENT@`). It never interrupts an active conversation.
- **Vietnamese voice** — Vector thinks and replies in Vietnamese, spoken through his own speaker via OpenAI TTS (his onboard English voice can't pronounce Vietnamese). Configurable in `.env`.
- **No microphone?** — just type to Vector in the UI box; everything else (brain, voice, vision, actions) works the same.

> See `PRD.md` and `IMPLEMENTATION_PLAN.md` for the full design, research notes, and hardware verification.

### Configure the brain

Copy `.env.example` to `.env` and set your key:

```sh
cp .env.example .env
# then edit .env:
#   OPENAI_API_KEY=sk-...
#   VECTOR_GPT_MODEL=gpt-5.5   # any vision-capable model; e.g. gpt-4.1 / gpt-4o if 5.5 isn't enabled
```

`run.sh` auto-loads `.env`. The model is swappable with one line — no code change.

## Setup

### wire-pod

Anki Servers are down. You need to first setup a local server using [wire-pod](https://github.com/kercre123/wire-pod).

### vector-sdk

Setup the sdk using my [fork](https://github.com/kingardor/vector-python-sdk).

### Object Detection (Raspberry Pi Compatible)

The project now uses **Hugging Face Transformers** with OWL-ViT for zero-shot object detection, optimized for Raspberry Pi CPU execution.

Install dependencies:
```sh
pip install -r requirements.txt
```

### Installation Steps

1. Install system dependencies (Debian/Ubuntu):
```sh
sudo apt-get update
sudo apt-get install -y python3-pip python3-dev libjpeg-dev zlib1g-dev
```

2. Install Python packages:
```sh
pip install -r requirements.txt
pip install -e vector-python-sdk/
```

## Run

Start everything (brain server + Vector's-mic voice bridge via wire-pod):

```sh
./run.sh
```

Then say **"Hey Vector, …(tiếng Việt)"**. Other modes:

```sh
./run.sh --no-pod   # brain server only (e.g. to test /stt and /v1)
./run.sh --sdk      # legacy standalone SDK app (app.py: text/USB-mic, agent loop, touch)
```

See `BRIDGE_SETUP.md` for the full architecture.

### Performance Notes

- First run will download the OWL-ViT model (~500MB)
- Detection speed on Raspberry Pi 4: ~2-5 seconds per frame
- For faster inference, consider using smaller images or the `google/owlvit-base-patch16` model