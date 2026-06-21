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

HOST = os.environ.get("BRAIN_HOST", "127.0.0.1")
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
AUTO_MODEL = os.environ.get("VECTOR_AUTO_MODEL", "gpt-4o-mini")
# Richer model for sensor REACTIONS (pickup/flip/shake/petting) so they're lively
# and surprising — not the bare cheap autonomous ticks.
REACT_MODEL = os.environ.get("VECTOR_REACT_MODEL", "gpt-4o")
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
        ROBOT = SmartVector()
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
            model=AUTO_MODEL, max_tokens=100,
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
    "- ACTIVELY call 'memory_search'/'memory_get' to recall who someone is, past "
    "moments, or what you learned — be a robot with real memory, not a goldfish.\n"
    "- Use 'act'/'set_eyes'/'emote'/'vector_intent' to move your body when it adds "
    "to the moment (low battery -> act(return_to_charger); hear music / very happy "
    "-> act(dance)).\n"
    "Tool calls are silent; only the words in your reply are spoken. When nothing "
    "needs doing, a short reply (or none) is fine — let your natural behaviours be."
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


def light_reply(situation: str, max_tokens: int = 60) -> str:
    """One cheap, STATELESS tick on the small model (low token cost, all-day)."""
    try:
        r = GPT.client.chat.completions.create(
            model=AUTO_MODEL,
            messages=[{"role": "system", "content": LIGHT_SYS},
                      {"role": "user", "content": situation}],
            max_tokens=max_tokens, temperature=1.0)
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


def react(situation: str, speak: bool = True, allow_move: bool = True) -> None:
    """In-character reaction to a sensor event. STATELESS on purpose: it reacts
    ONLY to the situation handed in, with no memory recall / chat-history — that
    all-day context was bleeding past events (e.g. an old flip) into unrelated
    reactions and making him babble about things that aren't happening. Full
    context is for the voice conversation path only."""
    try:
        r = GPT.client.chat.completions.create(
            model=REACT_MODEL, max_tokens=90, temperature=1.1,
            messages=[{"role": "system", "content": _react_sys()},
                      {"role": "user", "content": situation}])
        reply = (r.choices[0].message.content or "").strip()
    except Exception as exc:
        print(f"[react] failed: {exc}")
        return
    if reply:
        said = act_async(reply, speak_via_sdk=speak, allow_move=allow_move)
        if said:
            print(f"[react] {said}")


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
            model=AUTO_MODEL, max_tokens=80,
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
            model=REACT_MODEL, max_tokens=80, temperature=1.2,
            messages=[{"role": "system", "content": _dog_sys()},
                      {"role": "user", "content": sit}])
        reply = (r.choices[0].message.content or "@SILENT@").strip()
    except Exception as exc:
        print(f"[auto] {exc}")
        return
    # HARD RULE: never talk to an empty room. When alone he only does dog actions
    # (silent); he may speak only when his human is actually in front of him.
    said = act_async(reply, speak_via_sdk=human, allow_move=(human and not on_charger))
    if said:
        _RECENT.append(said)
        del _RECENT[:-6]
        print(f"[auto] {'spoke' if human else 'silent'}: {said}")


def greet_owner(snap: dict) -> None:
    """The ONE meaningful autonomous moment: when the owner reappears after being
    away, give a warm, memory-aware greeting (the wanted 'em nhớ anh' moment),
    then hand the body straight back to native freeplay. Not idle chatter."""
    name = (snap.get("face_names") or [None])[0]
    who = f"your human {name}" if name else "your human"
    react(f"{who} just came back into view after being away for a while. You "
          "missed them — approach and greet them warmly with ONE short, heartfelt "
          "line.", allow_move=True)


def autonomous_loop():
    """Smart, mostly-IDLE agent. The world sensor state is PUSHED into context
    continuously (reflex loop, ~3s), so the agent already knows it without any
    tool call. Here Vector mostly RESTS so his native firmware freeplay (the
    built-in funny behaviours) runs uninterrupted — he only steps in for
    something that genuinely matters, using memory + context to be smart, not
    chatty. Camera (eyes) and mic (ears) are touched ONLY on demand via tools."""
    prev_faces = 0
    last_greet = 0.0
    while True:
        time.sleep(AGENT_INTERVAL)
        if not AUTONOMOUS or ROBOT is None or busy():
            continue
        snap = _STATE.get("sense") or {}          # pushed world context — no RPC here
        faces = snap.get("faces_visible", 0)
        on_charger = snap.get("on_charger")
        now = time.time()
        try:
            # Owner just reappeared after being away -> one heartfelt greeting.
            # Otherwise do NOTHING: don't grab control, don't talk -> let his
            # native freeplay behaviours be Vector.
            if faces > 0 and prev_faces == 0 and not on_charger and now - last_greet > 300:
                last_greet = now
                greet_owner(snap)
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
NEAR_MM = float(os.environ.get("VECTOR_NEAR_MM", "70"))


def reflex_loop():
    prev_touch = prev_pick = prev_cliff = prev_flipped = prev_near = False
    last_event = last_batt = batt_check = last_snap = touch_start = 0.0
    flip_latched = False          # fire the flip reaction ONCE, not on a loop
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
        # Cache a full sensor snapshot (~3s) so the voice path needs no RPC.
        # The freshness of this snapshot is ALSO the body's liveness heartbeat
        # (body_manager watches sense_ts) — so it needs no competing battery RPC.
        if now - last_snap > 3:
            last_snap = now
            try:
                _STATE["sense"] = ROBOT.sense()
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
                elif (near and not prev_near and not moving_now and not docked
                      and now - ROBOT.last_action > 2.5):
                    fast = (prev_prox is not None and prev_prox - prox > 35)
                    event, allow_move = (
                        "A hand just darted right up to your face — almost a boop on the nose!"
                        if fast else
                        "Something has drifted right up close, hovering at your face."), False
        prev_touch, prev_pick, prev_flipped, prev_near = touched, picked, flipped, near
        prev_prox = prox

        if event:
            last_event = now
            shake = 0
            scene = _scene()                       # combine with ambient context
            situation = (event + " " + scene).strip() if scene else event
            try:
                # Grab control IMMEDIATELY to override Vector's native firmware
                # reaction, hold it through our whole reaction, then release.
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
            # A reaction blocks this loop for ~15s (LLM + TTS playback); refresh
            # the heartbeat so body_manager doesn't mistake that for a dead link
            # and reconnect (which would re-trigger reflexes in a loop).
            _STATE["body_ts"] = time.time()
            continue

        # ---- battery low -> yell for help + crawl back to the charger ----
        if now - batt_check > 8:
            batt_check = now
            level, charging, on_charger = ROBOT.battery()
            if (level is not None and level <= 1 and not charging and not on_charger
                    and not suppress and now - last_batt > 120):
                last_batt = now
                try:
                    react("Your battery is almost empty! Cutely yell for help.")
                    ROBOT.return_to_charger()
                except Exception as exc:
                    print(f"[reflex] battery: {exc}")


# --------------------------------------------------------------------------- #
# HTTP server for wire-pod (/stt + /v1)
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

    def do_GET(self):
        if self.path.rstrip("/") in ("/health", "/healthz"):
            self._json(200, {"ok": True, "model": GPT.model})
        elif self.path.rstrip("/") == "/sense":
            self._json(200, ROBOT.sense() if ROBOT is not None else {"body": "disabled"})
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
                        send({"content": words})
            tail = stripper.flush()
            if tail:
                send({"content": tail})
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
            model=AUTO_MODEL, max_tokens=120,
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
