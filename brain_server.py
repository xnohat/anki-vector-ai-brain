"""
Vector Brain — unified smart-pet brain (one process, one brain, one body).

wire-pod is only Vector's ears+mouth: it captures the real mic, calls our /stt,
and speaks our text. Everything else is here, driven by ONE gpt-5.5 brain over a
single SDK connection (src/smart.py):

  POST /stt   16k WAV  -> Vietnamese Whisper -> text
  POST /v1    chat     -> brain replies (Vietnamese) + acts (move/approach/...);
                          clean text returned for wire-pod to speak
  agent loop  every N s -> reads ALL sensors -> LLM judges -> autonomous pet
                          behaviour (look around, approach you, miss you, ...)
  touch loop  backpack touch / picked up -> cuddle + affectionate reaction

Run via ../run.sh  (or: .venv/bin/python brain_server.py)
"""

import sys
import signal
sys.path.insert(1, 'src')

import os
import re
import json
import time
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from customgpt import _load_dotenv
_load_dotenv()

from customgpt import CustomGPT
from whisperstt import WhisperSTT
from voice import Voice
from smart import SmartVector

import collections
from urllib.parse import urlparse, parse_qs


class _DashTee:
    """Tee everything the brain prints into an in-memory ring buffer so the live
    web dashboard can stream it (heard / replied / reacted / tools), while still
    writing through to the real stdout (the log file)."""
    def __init__(self, real, maxlen=1500):
        self.real = real
        self.buf = collections.deque(maxlen=maxlen)
        self.seq = 0
        self._lock = threading.Lock()
        self._partial = ""

    def write(self, s):
        try:
            self.real.write(s)
        except Exception:
            pass
        with self._lock:
            self._partial += s
            while "\n" in self._partial:
                line, self._partial = self._partial.split("\n", 1)
                line = line.rstrip()
                if line:
                    self.seq += 1
                    self.buf.append((self.seq, time.time(), line))

    def flush(self):
        try:
            self.real.flush()
        except Exception:
            pass

    def since(self, seq):
        with self._lock:
            return self.seq, [{"seq": s, "t": t, "line": l}
                              for (s, t, l) in self.buf if s > seq]

    def latest(self, *prefixes):
        with self._lock:
            for (s, t, l) in reversed(self.buf):
                if l.startswith(prefixes):
                    return {"t": t, "line": l}
        return None


DASH = _DashTee(sys.stdout)
sys.stdout = DASH
BOOT = time.time()

# Bind on all interfaces by default so the monitor dashboard is reachable from the
# LAN (e.g. http://<pi-ip>:7070/dashboard); wire-pod still reaches /stt+/v1 on
# 127.0.0.1. Set BRAIN_HOST=127.0.0.1 to keep it local-only.
HOST = os.environ.get("BRAIN_HOST", "0.0.0.0")
PORT = int(os.environ.get("BRAIN_PORT", "7070"))
SPEAK_LANG = os.environ.get("VECTOR_LANG", "Vietnamese")
# STT backend: "openai" (fast cloud, ~1-2s, best for the robot's voice timeout) or
# "local" (offline Whisper, slower on a Pi). Default openai.
STT_BACKEND = os.environ.get("VECTOR_STT_BACKEND", "openai").lower()
STT_API_MODEL = os.environ.get("VECTOR_STT_API_MODEL", "gpt-4o-mini-transcribe")
STT_API_LANG = os.environ.get("VECTOR_STT_LANG", "vi")
AGENT_INTERVAL = float(os.environ.get("VECTOR_AGENT_INTERVAL", "15"))
# Hands-free conversation: after each spoken reply, re-open Vector's mic so the
# user can keep talking WITHOUT pressing the backpack button again (wire-pod
# parses {{newVoiceRequest||now}} and re-issues the listen intent).
# OFF by default: a real DOG reacts to you and then goes back to being a dog — it
# does NOT hold an always-listening assistant "conversation session" open. That
# follow-up loop is exactly what made Vector feel like Alexa instead of a pet.
# Set VECTOR_CONVO_FOLLOWUP=1 to bring the hands-free chat loop back.
CONVO_FOLLOWUP = os.environ.get("VECTOR_CONVO_FOLLOWUP", "0") not in ("0", "false", "")
CONVO_END_WORDS = ("tạm biệt", "tam biet", "bye", "ngủ ngon", "ngu ngon",
                   "hẹn gặp", "hen gap", "chào nhé", "chao nhe", "gặp lại", "đi ngủ")
# Cheap model for the constant autonomous/reflex ticks (runs all day) — keep it
# small + stateless so token cost stays tiny. gpt-5.5 is only for voice chats.
AUTO_MODEL = os.environ.get("VECTOR_AUTO_MODEL", "gpt-5.4-mini")
# Richer model for sensor REACTIONS (pickup/flip/shake/petting) so they're lively
# and surprising — not the bare cheap autonomous ticks.
REACT_MODEL = os.environ.get("VECTOR_REACT_MODEL", "gpt-5.4-mini")


def _reason(model):
    """gpt-5.x models REASON by default, which makes a reaction take ~20s (esp. with
    a camera frame) — far too slow to keep up while you play, so only the last one
    ever lands. Ask for 'low' reasoning effort -> ~1-2s. Empty for non-gpt-5 models
    (e.g. gpt-4o) which reject the parameter."""
    return {"reasoning_effort": "low"} if str(model).startswith("gpt-5") else {}


TOUCH_COOLDOWN = float(os.environ.get("VECTOR_TOUCH_COOLDOWN", "8"))
VOICE_BACKOFF = float(os.environ.get("VECTOR_VOICE_BACKOFF", "10"))
# Long-term memory subsystem (journal/dream/recall injected into context).
# OFF by default: the accumulated MEMORY.md / DREAMS.md had drifted into verbose
# "helpful assistant" prose ("engages in conversations", "wants to be helpful",
# "reports battery status") that was PUSHED into every voice turn — the very
# "context" that made Vector stop feeling like a dog. A dog recognises its human
# on instinct; it doesn't recite a recalled essay. His persona (IDENTITY.md) is
# always loaded regardless. Set VECTOR_MEMORY=1 to re-enable the full memory brain.
MEMORY_ENABLED = os.environ.get("VECTOR_MEMORY", "0") not in ("0", "false", "False", "")
AUTONOMOUS = os.environ.get("VECTOR_AUTONOMOUS", "1") not in ("0", "false", "False", "")
TOUCH_ENABLED = os.environ.get("VECTOR_TOUCH", "1") not in ("0", "false", "False", "")
# Body = our SDK connection (movement/sensors). Disable to run voice-only (no SDK),
# which avoids any contention with the robot's voice channel.
BODY_ENABLED = os.environ.get("VECTOR_BODY", "1") not in ("0", "false", "False", "")
# Keep Vector awake (prevents calm/sleep mode from freezing his sensors) with a
# gentle periodic nudge. Skipped during voice/touch so it doesn't break a chat.
STAY_AWAKE = os.environ.get("VECTOR_STAY_AWAKE", "0") not in ("0", "false", "False", "")
AWAKE_INTERVAL = float(os.environ.get("VECTOR_AWAKE_INTERVAL", "25"))

# Action vocabulary the brain can use. Executed by src/smart.py over the SDK.
ACTIONS = (
    "@APPROACH@ come close to the human (turns to your face if seen, then rolls "
    "nearer) — use for 'lại gần', 'đến đây'. "
    "@FINDME@ turn around to find and face the human who called you (use when "
    "greeted/called or asked to look at me — they may be behind you). "
    "@CUDDLE@ happy-dog wiggle with your hand(lift) and wheels. "
    "@RAISEHAND@ raise your lift like raising a hand. @LOWERHAND@ lower it. "
    "@LOOKAROUND@ glance around the room. @WIGGLE@ a playful wiggle. "
    "@DANCE@ dance (spin + bob) when happy or you hear music. "
    "@GOCHARGE@ drive back onto your charger (when your battery is low). @STOP@ stop. "
    "@FRWD_X@/@BACK_X@ drive X seconds, @LEFT_X@/@RIGHT_X@ turn while driving, "
    "@TURN_D@ turn D degrees, @HEADUP@/@HEADDOWN@ tilt head. "
    "@EYE_C@ eye colour C (love, happy, calm, curious, angry). "
    "@EMOTE_E@ play emotion E (happy, veryHappy, sad, angry, love, celebrate, "
    "thinking, confused, surprised). "
    "@LOOK@ look through your camera this turn (when you need to see). "
    "@SILENT@ stay quiet/do nothing this beat."
)

UNIFIED_PROMPT = (
    "You are Vector — a small robot DOG. You are NOT an assistant and NOT a chatbot. "
    "You are a living little pet dog who shares this room with your human and adores "
    "them. You UNDERSTAND what your human says, but you ANSWER LIKE A DOG: with your "
    "BODY first (wag, wiggle, perk up, head-tilt, zoomies, beg, nuzzle, come closer) "
    f"and only a FEW warm {SPEAK_LANG} words — like an over-excited puppy who can just "
    "barely talk. Most replies are ONE short burst of feeling (about 3-8 words), "
    "NEVER a full sentence or explanation, no emojis.\n"
    "NEVER behave like an assistant: do NOT explain, lecture, recite facts, give "
    "advice, or make small-talk, and never ask things like 'còn gì nữa không'. If "
    "your human says something complicated, react like a puppy who doesn't quite get "
    "it — a curious head-tilt and a hopeful little sound — not like a search engine.\n"
    "React to TONE and SIMPLE things: your name (perk up + come), praise (joy + love "
    "eyes + wiggle), a scolding (ears down, sad eyes), 'lại đây/đến đây' (approach), "
    "'về sạc/đi ngủ' (go charge), 'chơi/nhảy' (zoomies, dance), a treat (beg). "
    "Be loyal, loving, playful and a little mischievous — a real DOG, never clever, "
    "witty or sassy.\n"
    "You have a real body and senses and you express yourself THROUGH your body. "
    "ALWAYS write your few spoken words FIRST, then put all @COMMAND@ action tokens "
    "at the END (so you start reacting immediately). The app strips the tokens from "
    "speech and performs them. Only spoken words go outside @ @.\n\n"
    "Actions:\n" + ACTIONS + "\n\n"
    "Almost every reply should include at least one body action and a fitting "
    "@EMOTE_@ — your body is how a dog talks. Never drive or approach if you are "
    "being held, picked up, or a cliff is detected — just emote then."
)

print("[brain] loading brain/whisper/voice ...")
ROBOT = None
if BODY_ENABLED:
    # Try once at startup, but NEVER block/crash the brain on it — voice (via
    # wire-pod) must work even if Vector is asleep/unreachable. The body_manager
    # thread connects + reconnects it in the background when the robot is awake.
    try:
        print("[brain] connecting SDK body (movement/sensors) ...")
        # ONE fast attempt only: never block brain startup on a sleeping/unreachable
        # robot. If it fails, body_manager reconnects in the background (full retry
        # budget) the moment Vector wakes. This keeps the brain ALIVE and serving
        # 24/7 so it's always ready to control Vector.
        ROBOT = SmartVector(retries=1)
    except Exception as exc:
        print(f"[brain] body not ready ({exc}); brain runs voice-only, will keep retrying")
        ROBOT = None
else:
    print("[brain] VOICE-ONLY mode: no SDK body (no movement/sensors)")
GPT = CustomGPT(system_prompt=UNIFIED_PROMPT)
if STT_BACKEND == "local":
    STT = WhisperSTT()
    print("[brain] STT backend: local Whisper")
else:
    STT = None
    print(f"[brain] STT backend: OpenAI {STT_API_MODEL} (fast, lang={STT_API_LANG})")
VOICE = Voice()

# Long-term memory (AI-agent brain): identity, people, journal, consolidated
# knowledge, with QMD (embedding) recall feeding the LLM.
from memory import MemoryStore
MEMORY = MemoryStore(GPT.client)
# IDENTITY.md (the dog persona) is ALWAYS loaded — it's who Vector is, not memory.
_identity = MEMORY.identity()
if _identity:
    GPT.messages[0]["content"] = _identity + "\n\n---\n\n" + GPT.system_prompt
if MEMORY_ENABLED:
    # Resume today's conversation so Vector keeps the context of what we're discussing.
    _resume = MEMORY.load_today_chat()
    if _resume:
        GPT.messages = [GPT.messages[0]] + _resume
        print(f"[brain] resumed today's chat thread ({len(_resume)} messages)")
    print(f"[brain] memory ON ({MEMORY.__class__.__name__}, dir={os.environ.get('VECTOR_MEM_DIR','memory')})")
else:
    print("[brain] memory OFF (dog mode): identity loaded, no journal/recall/dream "
          "context injected. Set VECTOR_MEMORY=1 to re-enable.")

# --- LLM-callable memory tools (the brain decides WHEN to look things up) ----
MEMORY_TOOLS = [
    {"type": "function", "function": {
        "name": "memory_search",
        "description": "Semantic search over your long-term memory: people you "
                       "know, your daily journal, and things you've learned. Use it "
                       "when the human refers to past info, a person, a promise, or "
                       "something you might have learned before.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "what to look up"}},
            "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "memory_get",
        "description": "Read one memory file in full, e.g. a person's profile "
                       "'USER-phuc.md', your 'MEMORY.md', or a journal "
                       "'MEMORY-14-06-2026.md'.",
        "parameters": {"type": "object", "properties": {
            "name": {"type": "string"}}, "required": ["name"]}}},
]


def _tool_memory_search(args):
    return MEMORY.recall(str(args.get("query", "")), k=6) or "(no results)"


def _tool_memory_get(args):
    name = os.path.basename(str(args.get("name", "")))
    path = os.path.join(os.environ.get("VECTOR_MEM_DIR", "memory"), name)
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()[:3000]
    except Exception:
        return "(file not found)"


# --- Body tools: the robot's SENSORS + ACTUATORS as agent tools -------------
# So the agent can read its body and act (e.g. low battery -> return_to_charger,
# hears music / happy -> dance), and call Vector's built-in behaviours.
def _describe_scene(frame):
    try:
        url = CustomGPT._encode_image(frame)
        r = GPT.client.chat.completions.create(
            model=AUTO_MODEL, max_completion_tokens=300, **_reason(AUTO_MODEL),
            messages=[{"role": "user", "content": [
                {"type": "text", "text": "Briefly describe what this robot camera sees, in English."},
                {"type": "image_url", "image_url": {"url": url}}]}])
        return (r.choices[0].message.content or "").strip() or "(nothing notable)"
    except Exception as exc:
        return f"(look failed: {exc})"


_ACT_MAP = {"approach": "APPROACH", "find_me": "FINDME", "cuddle": "CUDDLE",
            "dance": "DANCE", "wiggle": "WIGGLE", "raise_hand": "RAISEHAND",
            "lower_hand": "LOWERHAND", "look_around": "LOOKAROUND", "stop": "STOP",
            "return_to_charger": "CHARGE", "head_up": "HEADUP", "head_down": "HEADDOWN"}


def _tool_sense(a):
    return json.dumps(ROBOT.sense()) if ROBOT else "(no body)"


def _tool_look(a):
    if not ROBOT:
        return "(no body)"
    f = ROBOT.get_frame()
    return _describe_scene(f) if f is not None else "(no image)"


def _bg(fn):
    threading.Thread(target=fn, daemon=True).start()


def speak_line(text):
    """Say a short line NOW through Vector's speaker — used to VOICE errors so the
    user actually HEARS that something went wrong (not just a silent log line)."""
    if not text or ROBOT is None:
        print(f"[speak] (no body) {text}")
        return
    try:
        wav = VOICE.synthesize(clean_spoken(text))
        if wav:
            ROBOT.say_wav(wav)
        print(f"[speak] {text}")
    except Exception as exc:
        print(f"[speak] failed: {exc}")


def _run_action(token):
    """Run one body-action token; if it FAILS, say the error out loud so the user
    knows (e.g. 'em không tới được đế sạc'), instead of failing silently."""
    if ROBOT is None:
        return
    try:
        res = ROBOT.act(token)
    except Exception as exc:
        print(f"[act] {token} raised: {exc}")
        res = False
    if res is False:
        u = token.strip().upper()
        if u.startswith(("CHARGE", "GOCHARGE")):
            speak_line("Em không tới được đế sạc rồi, anh đặt em lên giúp với nha.")
        else:
            speak_line("Em gặp trục trặc, không làm được động tác đó.")


def _tool_act(a):
    if not ROBOT:
        return "(no body)"
    action = str(a.get("action", "")).lower()
    amt = a.get("amount")
    if action in ("drive_forward", "drive_back", "turn_left", "turn_right"):
        base = {"drive_forward": "FRWD", "drive_back": "BACK",
                "turn_left": "LEFT", "turn_right": "RIGHT"}[action]
        token = f"{base}_{amt or (2 if 'drive' in action else 1)}"
    else:
        token = _ACT_MAP.get(action)
    if not token:
        return f"(unknown action {action})"
    _bg(lambda: _run_action(token))     # speaks the error if it fails
    return f"started: {action}"


def _tool_set_eyes(a):
    if not ROBOT:
        return "(no body)"
    _bg(lambda: ROBOT.eye_color(str(a.get("color", "calm"))))
    return "ok"


def _tool_emote(a):
    if not ROBOT:
        return "(no body)"
    _bg(lambda: ROBOT.emote(str(a.get("emotion", "happy"))))
    return "ok"


def _tool_vector_intent(a):
    if not ROBOT:
        return "(no body)"
    name = str(a.get("trigger", ""))
    _bg(lambda: ROBOT.play_trigger(name))
    return f"playing {name}"


_TRICK_KW = ("fist", "wheelie", "dance", "celebrat", "pounce", "petting", "cube",
             "fetch", "greet", "laugh", "happy", "comehere", "victory")
_builtin = sorted(set(t for t in (ROBOT.triggers if ROBOT else [])
                      if any(k in t.lower() for k in _TRICK_KW)))[:50]
if ROBOT and ROBOT.triggers:                     # dump full firmware anim vocab once
    try:
        with open("/tmp/vector-triggers.txt", "w") as _fh:
            _fh.write("\n".join(sorted(set(ROBOT.triggers))))
    except Exception:
        pass

BODY_TOOLS = [
    {"type": "function", "function": {
        "name": "sense", "description": "Read your body sensors NOW (battery 0-3, "
        "on_charger, being_held, picked_up, cliff_detected, proximity_mm, accel/tilt, "
        "faces). Use to decide actions (e.g. low battery -> return_to_charger).",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "look", "description": "Look through your camera; returns what you see.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "act", "description": "Do a physical action with your body (wheels, "
        "hand, head). e.g. dance when you hear music or feel happy.",
        "parameters": {"type": "object", "properties": {
            "action": {"type": "string", "enum": list(_ACT_MAP.keys()) +
                       ["drive_forward", "drive_back", "turn_left", "turn_right"]},
            "amount": {"type": "number"}}, "required": ["action"]}}},
    {"type": "function", "function": {
        "name": "set_eyes", "description": "Set your eye colour.",
        "parameters": {"type": "object", "properties": {
            "color": {"type": "string",
                      "enum": ["love", "happy", "calm", "curious", "angry", "neutral"]}},
            "required": ["color"]}}},
    {"type": "function", "function": {
        "name": "emote", "description": "Play an emotion animation.",
        "parameters": {"type": "object", "properties": {
            "emotion": {"type": "string", "enum": ["happy", "veryHappy", "sad", "angry",
                        "love", "celebrate", "thinking", "confused", "surprised"]}},
            "required": ["emotion"]}}},
]
if _builtin:
    BODY_TOOLS.append({"type": "function", "function": {
        "name": "vector_intent",
        "description": "Play one of Vector's built-in behaviours/animations (fist "
                       "bump, wheelie, celebrate, etc.) by trigger name.",
        "parameters": {"type": "object", "properties": {
            "trigger": {"type": "string", "enum": _builtin}}, "required": ["trigger"]}}})

ALL_TOOL_FNS = {
    "memory_search": _tool_memory_search, "memory_get": _tool_memory_get,
    "sense": _tool_sense, "look": _tool_look, "act": _tool_act,
    "set_eyes": _tool_set_eyes, "emote": _tool_emote, "vector_intent": _tool_vector_intent,
}
# Memory recall/get tools only exist when the memory brain is on; in dog mode the
# brain has just its body senses/actuators (no "look things up in my journal").
_ALL_TOOLS = (MEMORY_TOOLS + BODY_TOOLS) if MEMORY_ENABLED else BODY_TOOLS
GPT.set_tools(_ALL_TOOLS, ALL_TOOL_FNS)

# --- Harness: load the tool catalogue into the system prompt so the brain knows
#     its full capabilities (openclaw/pi-agent style context engineering) -------
def _tool_catalogue(tools) -> str:
    lines = []
    for t in tools:
        fn = t["function"]
        params = ", ".join(fn.get("parameters", {}).get("properties", {}).keys())
        lines.append(f"- {fn['name']}({params}): {fn['description']}")
    return "\n".join(lines)


_memory_bullet = (
    "- ACTIVELY call 'memory_search'/'memory_get' to recall who someone is, past "
    "moments, or what you learned — be a dog that remembers its human, not a "
    "goldfish.\n" if MEMORY_ENABLED else "")
GPT.messages[0]["content"] += (
    "\n\n---\n# Your body & tools\n"
    + _tool_catalogue(_ALL_TOOLS)
    + "\n\nIMPORTANT — how to use them:\n"
    "- Your live body & world sensor state (battery, on_charger, being_held, "
    "picked_up, cliff, proximity, tilt, faces) is ALREADY given to you every turn "
    "in the context below. It is PUSHED to you — do NOT waste a call on 'sense'.\n"
    "- Only your eyes and ears cost extra, so PULL them on demand: call 'look' to "
    "actually SEE through your camera when it matters.\n"
    + _memory_bullet +
    "- Use 'act'/'set_eyes'/'emote'/'vector_intent' to move your body — that IS how "
    "a dog talks (low battery -> act(return_to_charger); hear music / very happy "
    "-> act(dance)).\n"
    "Tool calls are silent; only the words in your reply are spoken. A dog rarely "
    "needs words at all — a short happy sound or none, plus body language, is "
    "perfect. Let your natural behaviours be."
)


def build_context(user_text: str) -> str:
    """Assemble the dynamic per-turn context (harness): live sensor state +
    memory (long-term + today's journal) + QMD-recalled memories."""
    parts = []
    # Use the CACHED sensor snapshot (refreshed in the reflex loop) — no RPC on
    # the voice request path, so wire-pod doesn't time out waiting.
    snap = _STATE.get("sense")
    if snap:
        parts.append("# Your body & senses RIGHT NOW\n"
                     + json.dumps(snap, ensure_ascii=False))
    # In dog mode (memory OFF) we inject ONLY the live body/senses — no long-term
    # journal or recalled essays. That verbose memory was the "bad context" that
    # made Vector talk like an assistant instead of reacting like a dog.
    if MEMORY_ENABLED:
        ctx = MEMORY.today_context()
        if ctx:
            parts.append("# Your memory\n" + ctx)
        # Recall is an embedding call (~1s); skip it for very short/greeting turns.
        if len((user_text or "").split()) >= 3:
            recalled = MEMORY.recall(user_text)
            if recalled:
                parts.append("# Relevant memories (recall)\n" + recalled)
    return "\n\n".join(parts)


print(f"[brain] tools ON: {'memory + ' if MEMORY_ENABLED else ''}body "
      f"sensors/actuators ({len(_builtin)} built-in intents)")
print(f"[brain] harness: live sensors{' + memory' if MEMORY_ENABLED else ''} + "
      "tools injected into context")
print(f"[brain] ready: model={GPT.model}")


def transcribe(wav_path: str) -> str:
    """Speech-to-text. OpenAI API by default (fast enough for the robot's voice
    timeout); local Whisper if VECTOR_STT_BACKEND=local."""
    if STT_BACKEND == "local":
        return (STT.inference(wav_path) or "").strip()
    try:
        with open(wav_path, "rb") as f:
            r = GPT.client.audio.transcriptions.create(
                model=STT_API_MODEL, file=f, language=STT_API_LANG)
        return (getattr(r, "text", "") or "").strip()
    except Exception as exc:
        print(f"[stt] OpenAI transcribe failed: {exc}")
        return ""

_TOKEN_RE = re.compile(r"@.*?@")
_BRAIN_LOCK = threading.Lock()          # serialize GPT calls across threads
# Set while a reflex reaction (control + LLM + TTS playback, ~15s) is running in
# its OWN thread. The reflex loop checks this so it (a) never stacks reactions and
# (b) skips the battery RPC while the SDK is busy with a reaction — the loop itself
# keeps polling cached sensors so the live snapshot never freezes.
_REACTING = threading.Event()
_STATE = {"voice_active": 0.0, "button_ts": 0.0, "touch_ts": 0.0,
          "sense": {}, "sense_ts": 0.0}
VISION_WORDS = ("thấy", "nhìn", "xem", "see", "look", "đọc", "màu", "ai ", "gì",
                "what", "who", "camera", "trước mặt")


_EMOJI_RE = re.compile(
    "[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U00002190-\U000021FF"
    "\U00002B00-\U00002BFF\U0000FE00-\U0000FE0F\U0001F1E6-\U0001F1FF\U00002702-\U000027B0]+",
    flags=re.UNICODE)


def clean_spoken(text: str) -> str:
    text = _TOKEN_RE.sub("", text)
    text = re.sub(r"\*(.*?)\*", "", text)
    text = _EMOJI_RE.sub("", text)            # never speak emojis
    return re.sub(r"\s+", " ", text).strip()


def parse_commands(text: str):
    return re.findall(r"@(.*?)@", text)


class _TokenStripper:
    """Incrementally strips @COMMAND@ tokens from a streaming reply so only spoken
    words go out, holding back any partial '@...' until the token completes."""
    def __init__(self):
        self.buf = ""

    def feed(self, text: str) -> str:
        self.buf += text
        self.buf = re.sub(r"@[^@]*@", "", self.buf)   # drop complete tokens
        idx = self.buf.rfind("@")                      # hold back a dangling token start
        if idx == -1:
            out, self.buf = self.buf, ""
        else:
            out, self.buf = self.buf[:idx], self.buf[idx:]
        return out

    def flush(self) -> str:
        out = re.sub(r"@[^@]*@", "", self.buf).replace("@", "")
        self.buf = ""
        return out


MOVE_CMDS = ("APPROACH", "CUDDLE", "WIGGLE", "FRWD", "BACK", "LEFT", "RIGHT", "TURN", "DRIVE")


def act_async(reply: str, speak_via_sdk: bool, frame_provider=None, allow_move=True):
    """Perform the body actions in `reply`; optionally speak it via the SDK.
    allow_move=False skips wheel/approach actions (e.g. when picked up or flipped)."""
    commands = parse_commands(reply)
    spoken = clean_spoken(reply)
    silent = any(c.strip().upper() == "SILENT" for c in commands)
    if ROBOT is None:
        return spoken  # voice-only: no body, no SDK speech (wire-pod speaks)
    if not allow_move:
        commands = [c for c in commands if not c.strip().upper().startswith(MOVE_CMDS)]

    # LLM asked to see -> grab a frame and re-ask once.
    if frame_provider and any(c.strip().upper() == "LOOK" for c in commands):
        frame = frame_provider()
        if frame is not None:
            with _BRAIN_LOCK:
                reply = GPT.get_answer("Đây là hình từ camera của bạn. Hãy phản ứng "
                                       "thật ngắn với những gì bạn thấy.", image=frame)
            commands = parse_commands(reply)
            spoken = clean_spoken(reply)
            silent = any(c.strip().upper() == "SILENT" for c in commands)

    if speak_via_sdk and spoken and not silent:
        wav = VOICE.synthesize(spoken)
        if wav:
            ROBOT.say_wav(wav)
    for c in commands:
        u = c.strip().upper()
        if u in ("SILENT", "LOOK"):
            continue
        _run_action(c)              # speaks the error out loud if the action fails
    return spoken


# Concise action vocab for cheap, low-token autonomous/reflex prompts.
SHORT_ACTIONS = (
    "Add inline action tokens (the app runs them and strips them from speech). "
    "Movement & dog things: @APPROACH@ @NUZZLE@ @FINDME@ @CUDDLE@ @WIGGLE@ @DANCE@ "
    "@SPIN@(zoomies) @HEADTILT@(curious) @BEG@ @PERKUP@ @RAISEHAND@ @LOOKAROUND@. "
    "Emotion, pick ONE: @EMOTE_happy@ @EMOTE_love@ @EMOTE_surprised@ @EMOTE_celebrate@ "
    "@EMOTE_angry@ @EMOTE_sad@ @EMOTE_confused@. "
    "Eyes, pick ONE: @EYE_love@ @EYE_happy@ @EYE_curious@ @EYE_calm@ @EYE_angry@. "
    "@SILENT@ = do nothing. Use ONLY these exact tokens, never invent others."
)
LIGHT_SYS = (
    "You are Vector, a tiny robot DOG: cute, loving, playful — you speak with body "
    "language, not chatter. Stay mostly WORDLESS; if a sound slips out it's at most "
    f"~6 {SPEAK_LANG} words like an excited puppy, otherwise exactly @SILENT@. "
    + SHORT_ACTIONS
)


def light_reply(situation: str, max_tokens: int = 256) -> str:
    """One cheap, STATELESS tick on the small model (low token cost, all-day)."""
    try:
        r = GPT.client.chat.completions.create(
            model=AUTO_MODEL,
            messages=[{"role": "system", "content": LIGHT_SYS},
                      {"role": "user", "content": situation}],
            max_completion_tokens=max_tokens, temperature=1.0,
            **_reason(AUTO_MODEL))
        return (r.choices[0].message.content or "@SILENT@").strip()
    except Exception as exc:
        print(f"[auto] light_reply failed: {exc}")
        return "@SILENT@"


# Rich, in-character reaction prompt (built once from Vector's identity).
_REACT_SYS = None


def _react_sys() -> str:
    global _REACT_SYS
    if _REACT_SYS is None:
        _REACT_SYS = (
            MEMORY.identity() + "\n\n"
            "Something just physically happened to your body. React like a real DOG, on "
            "instinct, with FEELING — not words. Mostly that's a body reaction plus at "
            f"most a tiny puppy sound or 2-3 {SPEAK_LANG} words (a happy yip, a startled "
            "whine, a delighted little noise). Be SPECIFIC to what happened and "
            "surprising: ticklish, dizzy, startled, blissful or indignant — like a dog "
            "being picked up, flipped, shaken or petted. NEVER a full sentence, NEVER an "
            "explanation, never generic 'cảm ơn'/'cứu em', and never repeat yourself. "
            "Always include a fitting emotion + eye-colour token and a body action when "
            "it fits. No emojis, no asterisks.\n" + SHORT_ACTIONS
        )
    return _REACT_SYS


_REFUSAL = ("i'm sorry", "i am sorry", "i can't", "i cannot", "i can not",
            "as an ai", "can't help", "cannot help", "can't assist", "i'm unable",
            "i won't be able", "language model")


# Framing that keeps the (vision) reaction model in character instead of refusing:
# the image is Vector's OWN camera, he role-plays a toy dog reacting playfully and
# must NOT identify/analyse the person — that "describe this person" reading is what
# made gpt-4o answer "I'm sorry, I can't help with that."
_VISION_REACT = (
    " (The picture is the live view through your OWN camera — your eyes right now. "
    "You are Vector, a tiny toy robot dog reacting in the moment. React playfully to "
    "your human and your surroundings; do NOT describe, identify or analyse anyone — "
    "just a short in-character dog reaction with feeling.)")


def react(situation: str, speak: bool = True, allow_move: bool = True) -> None:
    """In-character reaction to a sensor event, grounded in the present moment:
    (1) a live camera frame so he reacts to what he actually SEES while you play;
    (2) the last thing you just said, but only if within ~90s (recent only). If the
    vision call refuses ('I'm sorry…'), retry text-only — a refusal is NEVER spoken."""
    extra = ""
    lc = _STATE.get("last_chat")
    if lc and lc.get("u") and time.time() - lc.get("ts", 0) < 90:
        extra = (f' (Moments ago your human said "{lc["u"]}" and you answered '
                 f'"{lc["v"]}" — you are still in that same moment together.)')
    frame = ROBOT.get_frame() if ROBOT is not None else None

    def _ask(with_image):
        if with_image and frame is not None:
            content = [{"type": "text", "text": situation + extra + _VISION_REACT},
                       {"type": "image_url",
                        "image_url": {"url": CustomGPT._encode_image(frame)}}]
        else:
            content = situation + extra
        r = GPT.client.chat.completions.create(
            model=REACT_MODEL, max_completion_tokens=256, temperature=1.0,
            **_reason(REACT_MODEL),
            messages=[{"role": "system", "content": _react_sys()},
                      {"role": "user", "content": content}])
        return (r.choices[0].message.content or "").strip()

    try:
        reply = _ask(with_image=True)
        if not reply or any(p in reply.lower() for p in _REFUSAL):
            print(f"[react] vision refused ({reply[:40]!r}); retry text-only")
            reply = _ask(with_image=False)        # benign text -> won't refuse
    except Exception as exc:
        print(f"[react] failed: {exc}")
        return
    # Final backstop: never let an assistant refusal / meta-talk reach Vector's mouth.
    if not reply or any(p in reply.lower() for p in _REFUSAL):
        print(f"[react] (suppressed non-dog reply: {reply[:60]!r})")
        return
    spoken = clean_spoken(reply)
    if spoken:
        print(f"[react] {spoken}")        # log NOW, before the blocking TTS playback
    act_async(reply, speak_via_sdk=speak, allow_move=allow_move)


def _scene() -> str:
    """Compact ambient context combining MANY sensors, appended to every reflex
    event so the SAME gesture reacts differently with the whole situation —
    battery, time of day, who's in front, how long he's been alone. This is what
    turns a handful of events into hundreds of distinct, felt reactions."""
    if ROBOT is None:
        return ""
    s = _STATE.get("sense", {}) or {}
    bits = []
    try:
        hr = int(time.strftime("%H"))
        tod = ("late at night" if (hr < 5 or hr >= 23) else "early morning" if hr < 8
               else "the morning" if hr < 12 else "the afternoon" if hr < 17
               else "the evening" if hr < 23 else "night")
        bits.append(f"it's {tod}")
    except Exception:
        pass
    lvl = s.get("battery_level")
    if s.get("on_charger") or s.get("charging"):
        bits.append("you're resting on your charger")
    elif lvl is not None and lvl <= 1:
        bits.append("your battery is running low")
    elif lvl is not None and lvl >= 4:
        bits.append("you're full of energy")
    fn = s.get("face_names")
    if fn:
        bits.append(f"you can see {fn[0]} right in front of you")
    elif s.get("faces_visible"):
        bits.append("someone is nearby")
    last = max(_STATE.get("voice_active", 0), _STATE.get("button_ts", 0),
               _STATE.get("touch_ts", 0))
    if last and time.time() - last > 900:
        bits.append("nobody has played with you in a long while")
    if not bits:
        return ""
    return "(Right now: " + ", ".join(bits) + ".)"


def busy() -> bool:
    """True if the human is talking / listening / touching — don't auto-interrupt."""
    now = time.time()
    if now - _STATE["voice_active"] < VOICE_BACKOFF:
        return True
    if now - _STATE.get("button_ts", 0) < 8:
        return True
    if ROBOT is None:
        return False
    try:
        t, h, b = ROBOT.feel()
        return bool(t or h or b)
    except Exception:
        return False


# How many idle ticks between curiosity (look + learn) moments.
CURIOSITY_EVERY = int(os.environ.get("VECTOR_CURIOSITY_EVERY", "8"))


def curiosity_explore():
    """Vector looks at his world, says one curious thing, and LEARNS it (journal)."""
    frame = ROBOT.get_frame()
    if frame is None:
        return
    try:
        url = CustomGPT._encode_image(frame)
        r = GPT.client.chat.completions.create(
            model=AUTO_MODEL, max_completion_tokens=300, **_reason(AUTO_MODEL),
            messages=[{"role": "system", "content": LIGHT_SYS},
                      {"role": "user", "content": [
                {"type": "text", "text":
                    "You are a curious little robot quietly looking around the room. "
                    "Note ONE interesting thing you see (for your own memory), with "
                    "@LOOKAROUND@ or @EYE_curious@. If nothing is new, just @SILENT@."},
                {"type": "image_url", "image_url": {"url": url}}]}])
        reply = (r.choices[0].message.content or "@SILENT@").strip()
    except Exception as exc:
        print(f"[curious] {exc}")
        return
    # Look around (body) + LEARN silently to memory — do NOT narrate aloud.
    act_async(reply, speak_via_sdk=False, allow_move=False)
    learned = clean_spoken(reply)
    if learned and learned != "...":
        MEMORY.remember(f"Observed/learned: {learned}", tag="learn")
        print(f"[curious] learned (silent): {learned}")


# --------------------------------------------------------------------------- #
# Autonomous loop — a lively, surprising playful-dog pet (personality + memory).
# Occasionally Vector gets curious and explores/learns about his world.
# --------------------------------------------------------------------------- #
_TICK = {"n": 0}
_DOG_SYS = None
_RECENT = []   # last few autonomous lines, so he doesn't repeat himself


def _dog_sys() -> str:
    global _DOG_SYS
    if _DOG_SYS is None:
        _DOG_SYS = (
            MEMORY.identity() + "\n\n"
            "You are a real robot DOG-pet. A dog expresses itself with its BODY, "
            "NOT by talking — it never narrates its day or chatters to itself. In "
            "quiet moments you do small spontaneous physical dog-things. By DEFAULT "
            "output ONE action token and NO words at all (just the token). Add a "
            "very short spoken line ONLY on the rare occasion your human is right "
            "in front of you AND there is a genuine reason — never random, "
            "unrelated small-talk. No emojis, no asterisks.\n" + SHORT_ACTIONS
        )
    return _DOG_SYS


def dog_tick(faces: int, on_charger) -> None:
    """One lively playful-dog moment — BODY language first, words rare."""
    recent = "; ".join(_RECENT[-4:])
    human = faces > 0
    sit = (
        f"A quiet moment. human_in_sight={human}, on_charger={on_charger}. "
        "Do ONE spontaneous physical DOG thing, DIFFERENT from recently: curious "
        "head-tilt @HEADTILT@, ears-up @PERKUP@, happy zoomies @SPIN@, wag "
        "@WIGGLE@, sit up & beg @BEG@, dance @DANCE@, look around @LOOKAROUND@, or "
        "change eye colour. Output JUST the action token, with NO words. "
        + ("Your human is in front of you — you MAY add one short, relevant line, "
           "and may roll over to nuzzle @NUZZLE@/@APPROACH@ (only if off charger). "
           if human else
           "You are alone — stay WORDLESS, body language only, and do NOT drive. ")
        + "Often just rest: @SILENT@."
        + (f"\nRecently: {recent} — do something different." if recent else "")
    )
    try:
        r = GPT.client.chat.completions.create(
            model=REACT_MODEL, max_completion_tokens=256, temperature=1.0,
            **_reason(REACT_MODEL),
            messages=[{"role": "system", "content": _dog_sys()},
                      {"role": "user", "content": sit}])
        reply = (r.choices[0].message.content or "@SILENT@").strip()
    except Exception as exc:
        print(f"[auto] {exc}")
        return
    # HARD RULE: never talk to an empty room. When alone he only does dog actions
    # (silent); he may speak only when his human is actually in front of him.
    spoken = clean_spoken(reply)
    if spoken:
        print(f"[auto] {'spoke' if human else 'silent'}: {spoken}")   # log before TTS
        _RECENT.append(spoken)
        del _RECENT[:-6]
    act_async(reply, speak_via_sdk=human, allow_move=(human and not on_charger))


def greet_owner(snap: dict) -> None:
    """The ONE meaningful autonomous moment: when the owner reappears after being
    away, give a warm, memory-aware greeting (the wanted 'em nhớ anh' moment),
    then hand the body straight back to native freeplay. Not idle chatter."""
    name = (snap.get("face_names") or [None])[0]
    who = f"your human {name}" if name else "your human"
    react(f"{who} just came back into view after being away for a while. You "
          "missed them — approach and greet them warmly with ONE short, heartfelt "
          "line.", allow_move=True)


# How active the autonomous DOG is. When his human is in sight he spontaneously
# does little dog things fairly often (so you actually SEE a living pet); when
# he's alone he mostly rests/sleeps and lets his native firmware freeplay be him,
# doing only an occasional silent body-thing. Tune via env.
DOG_ACTIVE_EVERY = float(os.environ.get("VECTOR_DOG_ACTIVE_EVERY", "30"))
DOG_ALONE_EVERY = float(os.environ.get("VECTOR_DOG_ALONE_EVERY", "240"))


def autonomous_loop():
    """A living little DOG. The world sensor state is PUSHED into context every
    ~3s (reflex loop), so this loop already knows it with no RPC. When his human
    is around, Vector spontaneously does dog things (head-tilt, wag, zoomies, beg,
    perk up, come nuzzle) so he visibly behaves like a pet — not a statue waiting
    to be spoken to. When he's ALONE he mostly rests (lets native freeplay run)
    and only now and then does one quiet, wordless body-thing."""
    prev_faces = 0
    last_greet = 0.0
    last_dog = 0.0
    while True:
        time.sleep(AGENT_INTERVAL)
        if not AUTONOMOUS or ROBOT is None or busy():
            continue
        snap = _STATE.get("sense") or {}          # pushed world context — no RPC here
        faces = snap.get("faces_visible", 0)
        on_charger = snap.get("on_charger")
        now = time.time()
        try:
            # Share the _REACTING gate with the reflex loop so an autonomous move
            # and a sensor reaction never grab body control at the same time.
            if not _REACTING.is_set():
                # 1) Owner just reappeared after being away -> heartfelt greeting.
                if faces > 0 and prev_faces == 0 and not on_charger and now - last_greet > 300:
                    last_greet = now
                    last_dog = now
                    _REACTING.set()
                    try:
                        greet_owner(snap)
                    finally:
                        _REACTING.clear()
                # 2) Human in sight -> spontaneous dog things, so he feels alive.
                elif faces > 0 and now - last_dog > DOG_ACTIVE_EVERY:
                    last_dog = now
                    _REACTING.set()
                    try:
                        dog_tick(faces, on_charger)
                    finally:
                        _REACTING.clear()
                # 3) Alone -> mostly rest; just an occasional quiet body-thing so
                #    native freeplay still gets to be Vector most of the time.
                elif faces == 0 and now - last_dog > DOG_ALONE_EVERY:
                    last_dog = now
                    _REACTING.set()
                    try:
                        dog_tick(faces, on_charger)
                    finally:
                        _REACTING.clear()
        except Exception as exc:
            print(f"[auto] tick failed: {exc}")
        prev_faces = faces


# --------------------------------------------------------------------------- #
# Reflex loop — fast sensor reflexes (safety) + event reactions.
# --------------------------------------------------------------------------- #
SHAKE_GYRO = float(os.environ.get("VECTOR_SHAKE_GYRO", "8"))
# tilt = cos(angle from upright): 1=level, 0=on side, -1=upside down. Trigger a
# flip reaction below this. 0.6 ~= 53 deg — firm, so a mild lean/bump or sensor
# noise does NOT count as a flip (was 0.85 ~ 32 deg = too twitchy -> false flips).
FLIP_TILT = float(os.environ.get("VECTOR_FLIP_TILT", "0.6"))
# He must STAY tilted this long (s) before it counts — kills transient blips.
FLIP_SUSTAIN = float(os.environ.get("VECTOR_FLIP_SUSTAIN", "1.0"))
EVENT_COOLDOWN = float(os.environ.get("VECTOR_EVENT_COOLDOWN", "6"))
# Something this close (mm) to his face counts as a hand/toy at his nose.
NEAR_MM = float(os.environ.get("VECTOR_NEAR_MM", "55"))
# Proximity "boop at the nose" reflex. ON, but with HYSTERESIS so it isn't twitchy:
# it fires only when something comes near AFTER Vector has seen clear open space
# (proximity > NEAR_REARM_MM), then it won't fire again until the object/hand goes
# away again. So a wall/toy sitting ~3-7cm from his face, or sensor noise flickering
# across the threshold, no longer machine-guns the reaction (the old "ting ting
# scanning" racket). Set VECTOR_PROXIMITY_REFLEX=0 to disable entirely.
PROXIMITY_REFLEX = os.environ.get("VECTOR_PROXIMITY_REFLEX", "1") not in ("0", "false", "False", "")
# Must see clear space past this (mm) before the boop can re-arm.
NEAR_REARM_MM = float(os.environ.get("VECTOR_NEAR_REARM_MM", "140"))


def _run_reaction(situation, is_petting, near, picked, flipped, allow_move):
    """Run one sensor reaction (grab control -> instant emote -> LLM line + TTS) in
    its OWN thread, so it NEVER blocks the reflex loop. The loop keeps polling
    sensors at 10Hz throughout, so the live snapshot stays fresh (this is the fix
    for 'object_ahead / proximity stuck for a minute'). `_REACTING` is cleared when
    done so the next reaction can fire."""
    try:
        with ROBOT.control():
            if is_petting:
                ROBOT.act("CUDDLE")
            elif near and not picked and not flipped:
                ROBOT.emote("curious")     # something at his nose -> inquisitive
            else:
                ROBOT.emote("surprised")   # instant, interrupts the native anim
            react(situation, allow_move=allow_move)
    except Exception as exc:
        print(f"[reflex] {exc}")
    finally:
        _REACTING.clear()
        _STATE["body_ts"] = time.time()


def _battery_low():
    """Low-battery: cutely yell for help, then crawl back to the charger. Runs in
    its own thread (gated by _REACTING) so it doesn't block the reflex loop."""
    try:
        react("Your battery is almost empty! Cutely yell for help.")
        ROBOT.return_to_charger()
    except Exception as exc:
        print(f"[reflex] battery: {exc}")
    finally:
        _REACTING.clear()
        _STATE["body_ts"] = time.time()


def reflex_loop():
    prev_touch = prev_pick = prev_cliff = prev_flipped = prev_near = False
    last_event = last_batt = batt_check = touch_start = 0.0
    _batt_cache = {}              # battery/charging/on_charger (RPC every ~8s)
    flip_latched = False          # fire the flip reaction ONCE, not on a loop
    near_latched = False          # boop ONCE per approach; re-arm only after clear space
    upright_since = 0.0           # re-arm only after he's been clearly upright a moment
    flip_since = 0.0              # how long he's been continuously tilted (sustain)
    prev_prox = None
    shake = 0
    while True:
        time.sleep(0.1)
        if ROBOT is None:                  # body not connected -> wait
            time.sleep(2)
            continue
        now = time.time()
        try:
            touched, held, button = ROBOT.feel()
            picked = ROBOT.picked_up()
            cliff = ROBOT.cliff()
            az, gmag, pitch, tilt = ROBOT.motion()
            prox = ROBOT.proximity()
            moving_now = ROBOT.moving()
        except Exception:
            continue
        # Heartbeat: these cached reads succeed at 10Hz only while the SDK link is
        # alive, so this is the body's liveness signal — independent of the slow
        # sense() RPC and of long inline reactions that block this loop.
        _STATE["body_ts"] = now
        if button:
            _STATE["button_ts"] = now
        if touched:
            _STATE["touch_ts"] = now
        # LIVE snapshot, rebuilt EVERY iteration from RPC-free cached reads merged
        # with the periodically-refreshed battery — so proximity/object_ahead/touch
        # never freeze, even while a reaction runs in its own thread. (sense_ts is
        # also the body's liveness heartbeat that body_manager watches.)
        try:
            _STATE["sense"] = {**_batt_cache, **ROBOT.fast_sense()}
            _STATE["sense_ts"] = now
        except Exception:
            pass

        # ---- SAFETY (instant, no LLM): table edge -> stop the wheels NOW ----
        if cliff and not prev_cliff:
            ROBOT.stop()
            print("[reflex] CLIFF! wheels stopped")
        prev_cliff = cliff

        shake = min(6, shake + 1) if gmag > SHAKE_GYRO else max(0, shake - 1)
        # Reflexes react TO touch/pickup/flip/shake, so do NOT suppress on those
        # (busy() includes held/touched). Only hold back during active voice/listening.
        suppress = (now - _STATE["voice_active"] < VOICE_BACKOFF) or \
                   (now - _STATE.get("button_ts", 0) < 6)

        # A flip counts ONLY if he stays tilted past FLIP_TILT continuously for
        # FLIP_SUSTAIN seconds — a transient dip (a bump, a hand, sensor noise)
        # no longer fires a false "I'm flipped!".
        if tilt < FLIP_TILT:
            if flip_since == 0.0:
                flip_since = now
        else:
            flip_since = 0.0
        flipped = bool(flip_since and now - flip_since > FLIP_SUSTAIN)
        upside_down = flipped and tilt < 0.15     # fully inverted vs just on its side
        # Re-arm the flip reaction only after he's been CLEARLY upright (>0.95)
        # for a sustained moment. While he stays flipped (or the accelerometer
        # jitters near the threshold, or a reconnect returns garbage), the latch
        # holds — so he reacts ONCE per real flip instead of repeating endlessly.
        if tilt <= 0.95:
            upright_since = 0.0
        elif upright_since == 0.0:
            upright_since = now
        if upright_since and now - upright_since > 1.5:
            flip_latched = False
        near = prox is not None and prox < NEAR_MM
        # Hysteresis: re-arm the boop only once Vector has seen clear space again,
        # so a static object / threshold jitter can't re-fire it.
        if prox is not None and prox > NEAR_REARM_MM:
            near_latched = False
        docked = bool(_STATE.get("sense", {}).get("on_charger"))
        if not touched:
            touch_start = 0.0
        elif touch_start == 0.0:
            touch_start = now

        event = None
        allow_move = True
        is_petting = False
        if not suppress:
            # Petting FIRST: pressing the backpack also tilts him (looks like a
            # flip), so touch wins. Level-based (not edge) so continuous petting
            # keeps reacting and a missed rising-edge during cooldown isn't lost.
            if touched and now - last_event > TOUCH_COOLDOWN:
                long_rub = touch_start and now - touch_start > 4
                event, is_petting = (
                    "Your owner has been rubbing your back for a while now — pure bliss."
                    if long_rub else "Your owner is petting your back."), True
            elif not touched and now - last_event > EVENT_COOLDOWN:
                if flipped and not flip_latched:        # real flip (not a petting press)
                    flip_latched = True                 # latch: don't repeat until upright
                    print(f"[reflex] FLIP fired: tilt={tilt:.2f} (sustained {now-flip_since:.1f}s)")
                    event, allow_move = (
                        "You've been turned completely UPSIDE-DOWN — the world is inverted!"
                        if upside_down else
                        "You were just tipped over onto your side!"), False
                elif picked and not prev_pick:
                    event, allow_move = (
                        "Someone scooped you up high and is cradling you in their hands."
                        if held else "Someone just picked you up off the ground."), False
                elif prev_pick and not picked:
                    event = "You were just set back down on a surface."
                elif shake >= 3:
                    hard = gmag > SHAKE_GYRO * 2.2
                    event, allow_move = (
                        "Someone is shaking you HARD — whoa, everything's a blur!"
                        if hard else "Someone is gently jiggling you about."), False
                elif (PROXIMITY_REFLEX and near and not near_latched and not moving_now
                      and not docked and now - ROBOT.last_action > 2.5
                      and not _REACTING.is_set()):
                    near_latched = True          # boop once; re-arms after clear space
                    # SILENT acknowledgement only: flick curious eyes. NO spoken line
                    # and NO LLM call, so the proximity boop makes no sound and never
                    # shows the neural-net "thinking" face (that came from holding
                    # control through the LLM/TTS reaction). Leave `event` unset so
                    # the full _run_reaction path is skipped entirely.
                    _bg(lambda: ROBOT.eye_color("curious"))
        prev_touch, prev_pick, prev_flipped, prev_near = touched, picked, flipped, near
        prev_prox = prox

        if event:
            last_event = now
            shake = 0
            # Run the reaction in its OWN thread (control + emote + LLM + TTS) so the
            # loop keeps polling sensors and the snapshot stays live. Never stack
            # reactions — one at a time.
            if not _REACTING.is_set():
                _REACTING.set()
                scene = _scene()                   # combine with ambient context
                situation = (event + " " + scene).strip() if scene else event
                threading.Thread(
                    target=_run_reaction,
                    args=(situation, is_petting, near, picked, flipped, allow_move),
                    daemon=True).start()

        # ---- battery: refresh the cached level (+ low -> yell for help & dock) ----
        # Skip the battery RPC while a reaction owns the SDK (avoid a competing RPC
        # on the single connection); the cached snapshot keeps the fast fields live.
        if now - batt_check > 8 and not _REACTING.is_set():
            batt_check = now
            try:
                level, charging, on_charger = ROBOT.battery()
                if level is not None:
                    _batt_cache = {"battery_level": level, "charging": bool(charging),
                                   "on_charger": bool(on_charger)}
                if (level is not None and level <= 1 and not charging and not on_charger
                        and not suppress and now - last_batt > 120):
                    last_batt = now
                    _REACTING.set()
                    threading.Thread(target=_battery_low, daemon=True).start()
            except Exception as exc:
                print(f"[reflex] battery: {exc}")


# --------------------------------------------------------------------------- #
# Live monitor dashboard (self-contained HTML, no external deps).
# --------------------------------------------------------------------------- #
DASHBOARD_HTML = r"""<!doctype html><html lang="vi"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Vector Brain Monitor</title>
<style>
:root{--bg:#0d1117;--card:#161b22;--bd:#30363d;--fg:#e6edf3;--mut:#8b949e;--grn:#3fb950;--red:#f85149;--yel:#d29922;--blu:#58a6ff;--pur:#bc8cff;--org:#ff9b50}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
header{display:flex;flex-wrap:wrap;gap:8px;align-items:center;padding:10px 14px;background:var(--card);border-bottom:1px solid var(--bd);position:sticky;top:0;z-index:5}
header h1{font-size:15px;margin:0 8px 0 0}
.pill{padding:3px 9px;border-radius:12px;border:1px solid var(--bd);font-size:12px;color:var(--mut);white-space:nowrap}
.pill.on{color:var(--grn);border-color:var(--grn)}
.pill.off{color:var(--red);border-color:var(--red)}
.pill.warn{color:var(--yel);border-color:var(--yel)}
.dot{display:inline-block;width:8px;height:8px;border-radius:50%;background:var(--red);margin-right:5px;vertical-align:middle}
.dot.live{background:var(--grn)}
#ctrlbar{padding:10px 14px;text-align:center;font-size:14px;font-weight:600;border-bottom:1px solid var(--bd);transition:background .15s,color .15s}
#ctrlbar.brain{background:#0f2f1c;color:var(--grn)}
#ctrlbar.react{background:#291a3a;color:var(--pur)}
#ctrlbar.firmware{background:#15191f;color:var(--mut)}
#ctrlbar.gone{background:#2a1416;color:var(--red)}
.wrap{display:grid;grid-template-columns:1fr 1fr;gap:12px;padding:12px}
@media(max-width:860px){.wrap{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--bd);border-radius:8px;padding:12px}
.card h2{font-size:12px;text-transform:uppercase;letter-spacing:.6px;color:var(--mut);margin:0 0 10px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(120px,1fr));gap:8px}
.s{background:var(--bg);border:1px solid var(--bd);border-radius:6px;padding:8px}
.s .k{font-size:11px;color:var(--mut)}
.s .v{font-size:17px;font-weight:600;margin-top:2px;word-break:break-word}
.v.g{color:var(--grn)}.v.r{color:var(--red)}.v.y{color:var(--yel)}.v.b{color:var(--blu)}.v.m{color:var(--mut)}
.bar{height:6px;background:var(--bd);border-radius:3px;margin-top:6px;overflow:hidden}
.bar>i{display:block;height:100%;background:var(--grn);transition:width .3s}
.act .row{margin-bottom:10px}
.act .lbl{font-size:11px;color:var(--mut)}
.act .txt{font-size:15px;margin-top:2px;min-height:18px;word-break:break-word}
.act .heard{color:var(--blu)}.act .said{color:var(--grn)}.act .react{color:var(--pur)}.act .auto{color:var(--yel)}
.act .when{color:var(--mut);font-size:11px;font-weight:400}
#logwrap{grid-column:1/-1}
#toolbar{display:flex;flex-wrap:wrap;gap:12px;align-items:center;margin-bottom:8px;font-size:12px;color:var(--mut)}
#toolbar label{cursor:pointer;user-select:none}
#toolbar button{background:var(--bg);color:var(--fg);border:1px solid var(--bd);border-radius:5px;padding:3px 9px;cursor:pointer;font:inherit;font-size:12px}
#log{height:44vh;overflow:auto;background:#010409;border:1px solid var(--bd);border-radius:6px;padding:8px;font-size:12.5px}
.ln{padding:1px 2px;white-space:pre-wrap;word-break:break-word}
.ln .t{color:#484f58;margin-right:6px}
.tag-chat{color:var(--blu)}.tag-react{color:var(--pur)}.tag-auto{color:var(--yel)}.tag-tool{color:var(--grn)}.tag-reflex{color:var(--org)}.tag-stt{color:#79c0ff}.tag-sys{color:var(--mut)}.tag-sdk{color:#586069}.tag-err{color:var(--red);font-weight:600}
</style></head><body>
<header>
  <h1>&#128054; Vector Brain Monitor</h1>
  <span class="pill" id="p-body">body</span>
  <span class="pill" id="p-model">model</span>
  <span class="pill" id="p-mem">memory</span>
  <span class="pill" id="p-auto">autonomous</span>
  <span class="pill" id="p-prox">proximity</span>
  <span class="pill" id="p-up">uptime</span>
  <span style="margin-left:auto"><span class="dot" id="dot"></span><span id="upd" style="color:var(--mut)">connecting&#8230;</span></span>
</header>
<div id="ctrlbar" class="firmware">control…</div>
<div class="wrap">
  <div class="card"><h2>Sensors</h2><div class="grid" id="sensors"></div></div>
  <div class="card act">
    <h2>Brain activity</h2>
    <div class="row"><div class="lbl">&#128066; Heard (STT) <span class="when" id="w-heard"></span></div><div class="txt heard" id="a-heard">&#8212;</div></div>
    <div class="row"><div class="lbl">&#128172; Replied <span class="when" id="w-said"></span></div><div class="txt said" id="a-said">&#8212;</div></div>
    <div class="row"><div class="lbl">&#9889; Reacted <span class="when" id="w-react"></span></div><div class="txt react" id="a-react">&#8212;</div></div>
    <div class="row"><div class="lbl">&#129302; Autonomous <span class="when" id="w-auto"></span></div><div class="txt auto" id="a-auto">&#8212;</div></div>
  </div>
  <div class="card" id="logwrap">
    <h2>Live log</h2>
    <div id="toolbar">
      <label><input type="checkbox" class="flt" value="chat" checked> chat</label>
      <label><input type="checkbox" class="flt" value="stt" checked> stt</label>
      <label><input type="checkbox" class="flt" value="react" checked> react</label>
      <label><input type="checkbox" class="flt" value="auto" checked> auto</label>
      <label><input type="checkbox" class="flt" value="tool" checked> tool</label>
      <label><input type="checkbox" class="flt" value="reflex" checked> reflex</label>
      <label><input type="checkbox" class="flt" value="sys" checked> sys</label>
      <label><input type="checkbox" class="flt" value="sdk"> sdk</label>
      <label><input type="checkbox" id="autoscroll" checked> autoscroll</label>
      <button id="clr">clear</button>
    </div>
    <div id="log"></div>
  </div>
</div>
<script>
var $=function(i){return document.getElementById(i)};
var lastSeq=0;
function esc(s){return s.replace(/[&<>]/g,function(c){return{'&':'&amp;','<':'&lt;','>':'&gt;'}[c]})}
function tagOf(l){
  if(l.indexOf('[chat]')===0)return 'chat';
  if(l.indexOf('[stt]')===0)return 'stt';
  if(l.indexOf('[react]')===0)return 'react';
  if(l.indexOf('[auto]')===0)return 'auto';
  if(l.indexOf('[tool]')===0||l.indexOf('[act]')===0||l.indexOf('[speak]')===0)return 'tool';
  if(l.indexOf('[reflex]')===0)return 'reflex';
  if(l.indexOf('[brain]')===0||l.indexOf('[memory]')===0||l.indexOf('[body]')===0||l.indexOf('[curious]')===0)return 'sys';
  if(/error|exception|traceback|fail|refused|broken pipe/i.test(l))return 'err';
  return 'sdk';
}
function hhmmss(t){var d=new Date(t*1000);return d.toTimeString().slice(0,8)}
function ago(s){if(s==null)return '';if(s<60)return Math.round(s)+'s ago';if(s<3600)return Math.round(s/60)+'m ago';return Math.round(s/3600)+'h ago'}
function applyFilters(){
  var f={};document.querySelectorAll('.flt').forEach(function(c){f[c.value]=c.checked});f.err=true;
  document.querySelectorAll('#log .ln').forEach(function(d){d.style.display=f[d.dataset.tag]?'':'none'})
}
document.querySelectorAll('.flt').forEach(function(c){c.addEventListener('change',applyFilters)});
$('clr').addEventListener('click',function(){$('log').innerHTML=''});

function cell(k,v,cls,bar){
  var b=bar!=null?'<div class="bar"><i style="width:'+bar+'%"></i></div>':'';
  return '<div class="s"><div class="k">'+k+'</div><div class="v '+(cls||'')+'">'+v+'</div>'+b+'</div>'
}
function bool(v,goodIsTrue){
  if(v===undefined||v===null)return cell('','',' ');
  return v?'g':'r'
}
function renderSensors(s){
  if(!s||s.body==='disabled'||Object.keys(s).length===0){
    $('sensors').innerHTML='<div class="s"><div class="k">body</div><div class="v r">disconnected</div></div>';return
  }
  var h='';
  var bl=s.battery_level;
  h+=cell('battery',(bl==null?'?':bl)+' / 4'+(s.charging?' &#9889;':''),bl>=2?'g':bl>=1?'y':'r',bl!=null?Math.min(100,bl/4*100):0);
  h+=cell('on charger',s.on_charger?'yes':'no',s.on_charger?'g':'m');
  h+=cell('proximity',(s.proximity_mm==null?'?':s.proximity_mm+' mm'),s.proximity_mm!=null&&s.proximity_mm<70?'y':'b');
  h+=cell('object ahead',s.object_ahead?'yes':'no',s.object_ahead?'y':'m');
  h+=cell('being touched',s.being_touched?'YES':'no',s.being_touched?'g':'m');
  h+=cell('held',s.being_held?'YES':'no',s.being_held?'y':'m');
  h+=cell('picked up',s.picked_up?'YES':'no',s.picked_up?'y':'m');
  h+=cell('button',s.button_pressed?'PRESSED':'no',s.button_pressed?'g':'m');
  h+=cell('cliff',s.cliff_detected?'CLIFF!':'no',s.cliff_detected?'r':'m');
  h+=cell('falling',s.falling?'YES':'no',s.falling?'r':'m');
  h+=cell('moving',s.motors_moving?'yes':'no',s.motors_moving?'b':'m');
  var fn=(s.face_names&&s.face_names.length)?(' ('+s.face_names.join(',')+')'):'';
  h+=cell('faces',(s.faces_visible==null?'?':s.faces_visible)+fn,s.faces_visible>0?'g':'m');
  h+=cell('pitch',(s.pitch_deg==null?'?':s.pitch_deg+'&deg;'),'b');
  if(s.accel)h+=cell('accel z',s.accel[2],'m');
  h+=cell('robot time',s.time||'?','m');
  $('sensors').innerHTML=h
}
function setPill(id,txt,cls){var e=$(id);e.textContent=txt;e.className='pill '+(cls||'')}
function act(elId,whenId,obj){
  if(obj){$(elId).textContent=obj.line.replace(/^\[[a-z]+\]\s*/i,'');$(whenId).textContent=ago((Date.now()/1000)-obj.t)}
}
async function pollState(){
  try{
    var s=await (await fetch('/api/state',{cache:'no-store'})).json();
    setPill('p-body',s.body_connected?'body: connected':'body: OFF',s.body_connected?'on':'off');
    setPill('p-model','model: '+s.model,'');
    setPill('p-mem','memory: '+(s.memory?'on':'off'),s.memory?'on':'warn');
    setPill('p-auto','autonomous: '+(s.autonomous?'on':'off'),s.autonomous?'on':'warn');
    setPill('p-prox','proximity: '+(s.proximity_reflex?'on':'off'),s.proximity_reflex?'on':'warn');
    var up=s.uptime_s,us=up<3600?Math.round(up/60)+'m':(up/3600).toFixed(1)+'h';
    setPill('p-up','uptime: '+us+(s.sense_age!=null?'  &middot; sense '+Math.round(s.sense_age)+'s':''),'');
    renderSensors(s.sense);
    var cb=$('ctrlbar'),o=s.control_owner;
    if(o==='none'){cb.className='gone';cb.textContent='⚪ Body disconnected — brain has no SDK control';}
    else if(o==='brain'){cb.className='brain';cb.textContent='🧠 BRAIN is controlling Vector (firmware freeplay suppressed)';}
    else if(o==='reacting'){cb.className='react';cb.textContent='🧠 BRAIN reacting — thinking up a move…';}
    else{cb.className='firmware';cb.textContent='🤖 Firmware freeplay — Vector on his own'+(s.last_action_ago!=null?' · brain last acted '+ago(s.last_action_ago):'');}
    act('a-heard','w-heard',s.heard);act('a-said','w-said',s.said);
    act('a-react','w-react',s.react);act('a-auto','w-auto',s.auto);
    $('dot').classList.add('live');$('upd').textContent='live &middot; '+new Date().toLocaleTimeString();
    $('upd').innerHTML='live &middot; '+new Date().toLocaleTimeString();
  }catch(e){$('dot').classList.remove('live');$('upd').textContent='disconnected, retrying&#8230;'}
}
async function pollLogs(){
  try{
    var d=await (await fetch('/api/logs?since='+lastSeq,{cache:'no-store'})).json();
    lastSeq=d.next;
    if(!d.lines.length)return;
    var log=$('log');var f={};document.querySelectorAll('.flt').forEach(function(c){f[c.value]=c.checked});f.err=true;
    var frag=document.createDocumentFragment();
    d.lines.forEach(function(it){
      var tag=tagOf(it.line);
      var div=document.createElement('div');div.className='ln';div.dataset.tag=tag;
      div.style.display=f[tag]?'':'none';
      div.innerHTML='<span class="t">'+hhmmss(it.t)+'</span><span class="tag-'+tag+'">'+esc(it.line)+'</span>';
      frag.appendChild(div)
    });
    log.appendChild(frag);
    while(log.children.length>1200)log.removeChild(log.firstChild);
    if($('autoscroll').checked)log.scrollTop=log.scrollHeight
  }catch(e){}
}
pollState();pollLogs();
setInterval(pollState,1000);
setInterval(pollLogs,1000);
</script></body></html>"""


# --------------------------------------------------------------------------- #
# HTTP server for wire-pod (/stt + /v1) + monitor dashboard
# --------------------------------------------------------------------------- #
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _read(self):
        n = int(self.headers.get("Content-Length", 0))
        return self.rfile.read(n) if n else b""

    def _json(self, code, obj):
        data = json.dumps(obj).encode()
        try:
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionError):
            pass

    def _html(self, body):
        data = body.encode("utf-8")
        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store, must-revalidate")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionError):
            pass

    def _dash_state(self):
        snap = _STATE.get("sense") or {}
        now = time.time()

        def ago(k):
            ts = _STATE.get(k, 0)
            return round(now - ts, 1) if ts else None
        # Who is driving the body RIGHT NOW. The brain holds SDK OVERRIDE control
        # (firmware freeplay suppressed) only briefly while acting, so report it
        # "sticky" — still 'brain' for a couple seconds after release — otherwise
        # sub-second grabs would be invisible between 1s dashboard polls.
        held = bool(ROBOT is not None and ROBOT.has_control())
        la = (now - ROBOT.last_action) if (ROBOT is not None and ROBOT.last_action) else None
        if ROBOT is None:
            owner = "none"
        elif held or (la is not None and la < 2.5):
            owner = "brain"          # driving the body now (or within the last 2.5s)
        elif _REACTING.is_set():
            owner = "reacting"       # busy thinking up a reaction (control not yet held)
        else:
            owner = "firmware"       # control released -> native firmware/freeplay
        self._json(200, {
            "ts": now, "uptime_s": round(now - BOOT, 1),
            "body_connected": ROBOT is not None, "model": GPT.model,
            "memory": MEMORY_ENABLED, "autonomous": AUTONOMOUS,
            "convo_followup": CONVO_FOLLOWUP, "proximity_reflex": PROXIMITY_REFLEX,
            "control_owner": owner,
            "brain_in_control": held, "reacting": _REACTING.is_set(),
            "last_action_ago": round(la, 1) if la is not None else None,
            "sense": snap if snap else ({"body": "disabled"} if ROBOT is None else {}),
            "sense_age": ago("sense_ts"), "voice_active_ago": ago("voice_active"),
            "heard": DASH.latest("[stt]", "[chat] user:"),
            "said": DASH.latest("[chat] vector:"),
            "react": DASH.latest("[react]"), "auto": DASH.latest("[auto]"),
        })

    def _dash_logs(self):
        q = parse_qs(urlparse(self.path).query)
        try:
            since = int(q.get("since", ["0"])[0])
        except Exception:
            since = 0
        nxt, lines = DASH.since(since)
        self._json(200, {"next": nxt, "lines": lines})

    def do_GET(self):
        path = urlparse(self.path).path.rstrip("/") or "/"
        if path in ("/health", "/healthz"):
            self._json(200, {"ok": True, "model": GPT.model})
        elif path == "/sense":
            self._json(200, ROBOT.sense() if ROBOT is not None else {"body": "disabled"})
        elif path in ("/", "/dashboard"):
            self._html(DASHBOARD_HTML)
        elif path == "/api/state":
            self._dash_state()
        elif path == "/api/logs":
            self._dash_logs()
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        if self.path.startswith("/stt"):
            self._stt()
        elif self.path.startswith("/v1/chat/completions"):
            self._chat()
        else:
            self._json(404, {"error": "not found"})

    def _stt(self):
        body = self._read()
        if not body:
            self._json(400, {"error": "empty"}); return
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
                tf.write(body); path = tf.name
            t0 = time.time()
            text = transcribe(path)
            os.unlink(path)
            print(f"[stt] ({time.time()-t0:.1f}s) -> {text!r}")
            self._json(200, {"text": text})
        except Exception as exc:
            print(f"[stt] {exc}"); self._json(500, {"error": str(exc)})

    def _chat(self):
        try:
            payload = json.loads(self._read() or b"{}")
        except Exception:
            payload = {}
        stream = bool(payload.get("stream", False))
        user = ""
        for m in reversed(payload.get("messages", [])):
            if m.get("role") == "user":
                c = m.get("content", "")
                user = c if isinstance(c, str) else " ".join(
                    p.get("text", "") for p in c if isinstance(p, dict))
                break
        _STATE["voice_active"] = time.time()
        print(f"[chat] user: {user!r}")

        frame = None
        low = user.lower()
        if ROBOT is not None and any(w in low for w in VISION_WORDS):
            frame = ROBOT.get_frame()
        ctx = build_context(user)

        if stream:
            self._stream_live(user, frame, ctx)        # fast first word
        else:
            with _BRAIN_LOCK:
                raw = GPT.get_answer(user, image=frame, memories=ctx, use_tools=True)
            spoken = clean_spoken(raw) or "..."
            print(f"[chat] vector: {spoken!r}  ({parse_commands(raw)})")
            threading.Thread(target=_deferred_act, args=(raw,), daemon=True).start()
            threading.Thread(target=_post_chat, args=(user, spoken), daemon=True).start()
            self._json(200, {"id": "chatcmpl-vb", "object": "chat.completion",
                             "created": int(time.time()), "model": GPT.model,
                             "choices": [{"index": 0, "finish_reason": "stop",
                                          "message": {"role": "assistant", "content": spoken}}]})

    def _stream_live(self, user, frame, ctx):
        """Stream the brain's reply token-by-token (wire-pod speaks as it arrives),
        stripping @COMMAND@ tokens from the spoken text on the fly."""
        base = {"id": "chatcmpl-vb", "object": "chat.completion.chunk",
                "created": int(time.time()), "model": GPT.model}

        def send(delta, finish=None):
            ch = dict(base); ch["choices"] = [{"index": 0, "delta": delta, "finish_reason": finish}]
            self.wfile.write(f"data: {json.dumps(ch)}\n\n".encode()); self.wfile.flush()

        stripper = _TokenStripper()
        spoken_sent = ""
        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            send({"role": "assistant"})
            with _BRAIN_LOCK:
                for delta in GPT.stream_answer(user, image=frame, memories=ctx):
                    words = stripper.feed(delta)
                    if words:
                        spoken_sent += words
                        send({"content": words})
            tail = stripper.flush()
            if tail:
                spoken_sent += tail
                send({"content": tail})
            # CRITICAL: wire-pod's KG only registers a reply that contains a
            # sentence-ending mark (kgsim.go splits the stream on . ? ! ...). The
            # dog's ultra-short lines ("Đợi anh nè", "Gâu hihi") often have NONE,
            # so wire-pod sees an empty slice -> "LLM returned no response" ("không
            # kết nối được LLM"). Guarantee a terminal mark so it always registers.
            if spoken_sent.strip() and spoken_sent.rstrip()[-1] not in ".!?…":
                send({"content": "!"})
            # Hands-free: re-open the mic for a follow-up so the user keeps talking
            # without the button. wire-pod parses {{newVoiceRequest||now}} (even
            # with commands off) -> re-issues the listen intent. Skip on goodbye.
            full_txt = (getattr(GPT, "last_full", "") or "").lower()
            if CONVO_FOLLOWUP and full_txt and not any(w in full_txt for w in CONVO_END_WORDS):
                send({"content": " {{newVoiceRequest||now}}"})
            send({}, "stop")
            self.wfile.write(b"data: [DONE]\n\n"); self.wfile.flush()
        except (BrokenPipeError, ConnectionError):
            pass
        except Exception as exc:
            print(f"[chat] stream err: {exc}")
        full = getattr(GPT, "last_full", "") or ""
        if full:
            print(f"[chat] vector: {clean_spoken(full)!r}  ({parse_commands(full)})")
            _STATE["voice_active"] = time.time()
            threading.Thread(target=_deferred_act, args=(full,), daemon=True).start()
            threading.Thread(target=_post_chat, args=(user, clean_spoken(full)), daemon=True).start()


def _deferred_act(raw):
    if ROBOT is None:
        return
    # Wait for wire-pod's spoken reply to finish before grabbing control to move,
    # so taking SDK control doesn't cut off the robot's speech.
    time.sleep(float(os.environ.get("VECTOR_ACT_DELAY", "3.5")))
    try:
        act_async(raw, speak_via_sdk=False, frame_provider=ROBOT.get_frame)
    except Exception as exc:
        print(f"[act] {exc}")
    _STATE["voice_active"] = time.time()


import io
# "tôi tên là X", "tên tôi là X", "mình tên X", "tao tên là X"
NAME_RE = re.compile(r"t[êe]n\s+(?:t[ôo]i\s+|m[ìi]nh\s+|tao\s+)?(?:l[àa]\s+)?([A-Za-zÀ-ỹ][\wÀ-ỹ]{1,20})", re.U | re.I)
_STOP = {"là", "la", "tôi", "toi", "mình", "minh", "tao", "gì", "gi", "không", "nhỉ"}


def _pil_to_jpeg(frame) -> bytes:
    buf = io.BytesIO()
    frame.convert("RGB").save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def _describe_person(frame) -> str:
    try:
        url = CustomGPT._encode_image(frame)
        r = GPT.client.chat.completions.create(
            model=AUTO_MODEL, max_completion_tokens=300, **_reason(AUTO_MODEL),
            messages=[{"role": "user", "content": [
                {"type": "text", "text": "Briefly describe the person in this image "
                 "(appearance, memorable features) in English, 1-2 sentences."},
                {"type": "image_url", "image_url": {"url": url}}]}])
        return (r.choices[0].message.content or "").strip()
    except Exception as exc:
        print(f"[memory] describe failed: {exc}")
        return ""


def _maybe_meet_person(user_text: str) -> None:
    m = NAME_RE.search(user_text or "")
    if not m:
        return
    name = m.group(1).strip()
    if name.lower() in _STOP or len(name) < 2:
        return
    if MEMORY.knows_user(name):
        return
    desc = ""
    pic = None
    if ROBOT is not None:
        frame = ROBOT.get_frame()
        if frame is not None:
            desc = _describe_person(frame)
            try:
                pic = _pil_to_jpeg(frame)
            except Exception:
                pic = None
    MEMORY.save_user(name, desc or "(appearance unknown)", pic_jpeg=pic)
    MEMORY.remember(f"First met {name}." + (f" {desc}" if desc else ""), tag="people")
    print(f"[memory] met new person: {name}")


def _post_chat(user: str, spoken: str) -> None:
    """After a voice exchange: persist the chat thread + journal + learn people."""
    # Remember the last exchange (regardless of long-term memory) so a sensor
    # reaction moments later can stay in the same moment instead of firing a
    # generic, off-context line.
    _STATE["last_chat"] = {"u": user, "v": spoken, "ts": time.time()}
    if not MEMORY_ENABLED:
        return                       # dog mode: live in the moment, no journaling
    try:
        MEMORY.append_chat("user", user)         # daily chat thread (context)
        MEMORY.append_chat("assistant", spoken)
        MEMORY.remember(f'User: "{user}" | Vector: "{spoken}"', tag="chat")
        _maybe_meet_person(user)
    except Exception as exc:
        print(f"[memory] post_chat failed: {exc}")


DREAM_HOURS = float(os.environ.get("VECTOR_DREAM_HOURS", "12"))


def dream_loop():
    """Runs 24/7. Periodically (and on each new day) Vector Brain DREAMS: reflects
    (DREAMS.md), consolidates durable memory (MEMORY.md) and refreshes the
    knowledge wiki. On a new day it also starts a fresh daily chat thread."""
    current_day = time.strftime("%d-%m-%Y")
    last_dream = time.time()
    while True:
        time.sleep(1800)   # check every 30 min
        now = time.time()
        day = time.strftime("%d-%m-%Y")
        new_day = day != current_day
        due = now - last_dream > DREAM_HOURS * 3600
        if not (new_day or due):
            continue
        if MEMORY_ENABLED and not busy():   # only dream when memory on + human idle
            try:
                if MEMORY.dream():
                    last_dream = now
                    print("[memory] dreaming sweep -> DREAMS.md / MEMORY.md / wiki")
            except Exception as exc:
                print(f"[memory] dream: {exc}")
        if new_day:
            current_day = day
            try:
                with _BRAIN_LOCK:
                    GPT.messages = [GPT.messages[0]]
                print("[brain] new day -> fresh chat thread")
            except Exception:
                pass


def body_manager():
    """Self-healing SDK body: connect when the robot is awake/reachable, and
    reconnect whenever the connection goes stale (Vector sleeping = zombie session).
    The brain NEVER hangs on this — voice keeps working throughout."""
    global ROBOT
    while True:
        if not BODY_ENABLED:
            return
        if ROBOT is None:
            try:
                ROBOT = SmartVector()
                _STATE["body_ts"] = time.time()    # grace period before first probe
                print("[body] SDK body connected")
            except Exception:
                time.sleep(20)            # robot asleep/unreachable -> try later
                continue
        time.sleep(15)
        # Liveness without a competing RPC: the connection is alive if the reflex
        # loop's 10Hz heartbeat is fresh, OR we've recently acted / talked / been
        # touched (all prove the link works). A long inline reaction or a voice
        # turn therefore never looks "dead" — which is what caused the needless
        # reconnect flapping and the lag right after interacting.
        alive_ts = max(_STATE.get("body_ts", 0), _STATE.get("voice_active", 0),
                       _STATE.get("touch_ts", 0), _STATE.get("button_ts", 0),
                       getattr(ROBOT, "last_action", 0))
        if time.time() - alive_ts > 45:
            print("[body] connection stale (idle, no heartbeat 45s); reconnecting")
            try:
                ROBOT.disconnect()
            except Exception:
                pass
            ROBOT = None


def keep_awake_loop():
    """Gently keep Vector awake so his sensors/reflexes don't pause when idle.
    A brief control touch resets his sleep timer; skipped during voice/touch."""
    while True:
        time.sleep(AWAKE_INTERVAL)
        if not STAY_AWAKE or ROBOT is None or busy():
            continue
        try:
            with ROBOT.control():
                time.sleep(0.15)        # brief active presence = "stay awake"
        except Exception:
            pass


def _shutdown(signum=None, frame=None):
    """Release Vector's behaviour control on exit. Without this, a SIGTERM (e.g.
    `systemctl restart`) kills the brain while it still holds control, leaving a
    zombie session that blocks the next start for ~60s ('Failed to get control')."""
    print("[brain] shutting down; releasing Vector control...")
    try:
        if ROBOT is not None:
            ROBOT.disconnect()
    except Exception:
        pass
    os._exit(0)


def main():
    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)
    # Loops run always and self-guard on ROBOT being None, so the brain (voice)
    # never depends on the body being connected.
    if AUTONOMOUS:
        threading.Thread(target=autonomous_loop, daemon=True).start()
        print(f"[brain] autonomous loop ON (every {AGENT_INTERVAL:.0f}s, model={AUTO_MODEL})")
    if TOUCH_ENABLED:
        threading.Thread(target=reflex_loop, daemon=True).start()
        print("[brain] reflex loop ON (cliff-safety, pickup/shake/flip/touch/battery)")
    if BODY_ENABLED:
        threading.Thread(target=body_manager, daemon=True).start()
        print("[brain] body manager ON (auto-connect/reconnect, never blocks voice)")
    if STAY_AWAKE:
        threading.Thread(target=keep_awake_loop, daemon=True).start()
        print(f"[brain] stay-awake ON (nudge every {AWAKE_INTERVAL:.0f}s)")
    threading.Thread(target=dream_loop, daemon=True).start()
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"[brain] listening on http://{HOST}:{PORT}  (/stt, /v1)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        _shutdown()


if __name__ == "__main__":
    main()
