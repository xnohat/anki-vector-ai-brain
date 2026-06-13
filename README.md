# Vector Brain — an AI Agent with a physical body

> An Anki **Vector** robot turned into a real AI Agent: one **gpt-5.5** brain with
> a body (camera, mic, speaker, hand, wheels), live senses, reflexes, tools, and
> persistent memory. It listens through Vector's own mic, thinks and speaks in
> Vietnamese, sees, feels, remembers people, and acts — all in one process.

![vector](resources/vector.jpg)

## What it is

The **Brain** (`brain_server.py`) is the agent's mind; **Vector** is its body.

- **Listen** — Vector's *real* microphone (press the backpack button or "Hey
  Vector") → wire-pod → our fast cloud STT.
- **Think** — one gpt-5.5 brain (Vietnamese, a curious sassy-but-loving pet).
- **Speak** — Vietnamese, streamed token-by-token so the first word comes fast.
- **See** — its camera (describe / read / recognize & photograph people).
- **Act** — approach you, cuddle (hand + wheels), find-and-face you, dance, raise
  a hand, drive/turn, eye colour, emote, and 50 built-in Vector animations.
- **Reflexes** — feels pickup / shake / flip / petting, **stops at a table edge**,
  and when the battery is low it yells for help and crawls back to the charger.
- **Be autonomous** — a cheap loop every 15s lets it judge its world and do
  delightful unexpected things; it's curious and journals what it learns.
- **Remember (AI Agent)** — identity, people (`USER-*` + photo), a daily journal,
  a daily chat thread for context, consolidated long-term memory, a knowledge
  wiki, and **QMD** (embedding) recall feeding the brain. It even "dreams":
  nightly it reflects, consolidates and refreshes its knowledge.
- **Tools** — the agent calls tools to sense and act: `memory_search`,
  `memory_get`, `sense`, `look`, `act`, `set_eyes`, `emote`, `vector_intent`.

## Architecture

```
 "Hey Vector / button" ─▶ Vector mic ─▶ wire-pod (VAD)
                                          │  POST /stt  (fast cloud STT)
                                          │  POST /v1   (gpt-5.5, streaming)
                                          ▼
                                   brain_server.py  ── SDK ──▶  Vector body
                                   (harness: identity + tools +      (move, hand,
                                    live sensors + memory + chat)      wheels, eyes)
                                          │
                                   memory/  (IDENTITY, USER-*, journal,
                                             chat thread, MEMORY.md, wiki, DREAMS)
```

wire-pod is just the ears+mouth bridge; the Brain does everything else. See
`BRIDGE_SETUP.md`, `PRD.md`, `IMPLEMENTATION_PLAN.md` for the full design.

## Setup

1. **wire-pod** — a local Vector server: <https://github.com/kercre123/wire-pod>
   (pair Vector, set your OpenAI key in its web UI).
2. **Python deps**
   ```sh
   python3 -m venv .venv && . .venv/bin/activate
   pip install -r requirements.txt
   pip install -e vector-python-sdk/
   ```
3. **Config** — `cp .env.example .env` and set `OPENAI_API_KEY`. Everything else
   (model, language, voice, intervals) has sensible defaults in `.env`.

## Run

```sh
./run.sh             # installs the wire-pod bridge, starts the brain + bridge
./run.sh --no-pod    # brain server only (for testing /stt and /v1)
```

Then press the backpack button and **speak Vietnamese**. The agent runs 24/7.

## Open source

All code, files and memory markers are in **English** so contributors can join
(Vector still *speaks* Vietnamese). Personal memory (journals, people, photos,
chat threads, embedding index) is gitignored — only the `IDENTITY.md` seed ships.
