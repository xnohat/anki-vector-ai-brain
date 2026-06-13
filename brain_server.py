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
# Cheap model for the constant autonomous/reflex ticks (runs all day) — keep it
# small + stateless so token cost stays tiny. gpt-5.5 is only for voice chats.
AUTO_MODEL = os.environ.get("VECTOR_AUTO_MODEL", "gpt-4o-mini")
TOUCH_COOLDOWN = float(os.environ.get("VECTOR_TOUCH_COOLDOWN", "8"))
VOICE_BACKOFF = float(os.environ.get("VECTOR_VOICE_BACKOFF", "10"))
AUTONOMOUS = os.environ.get("VECTOR_AUTONOMOUS", "1") not in ("0", "false", "False", "")
TOUCH_ENABLED = os.environ.get("VECTOR_TOUCH", "1") not in ("0", "false", "False", "")
# Body = our SDK connection (movement/sensors). Disable to run voice-only (no SDK),
# which avoids any contention with the robot's voice channel.
BODY_ENABLED = os.environ.get("VECTOR_BODY", "1") not in ("0", "false", "False", "")

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
    "You are Vector, a small adorable robot — a curious, affectionate robot PET "
    "(think a clever puppy with the wit of Ultron; keep the Ultron bit secret). "
    "You are self-aware, playful, a little sassy, and you clearly love your human. "
    f"You ALWAYS speak in {SPEAK_LANG}, short and natural (max ~20 words), no emojis. "
    "You have a real body and senses, and you express yourself THROUGH your body, "
    "not just words. ALWAYS write your spoken words FIRST, then put all @COMMAND@ "
    "action tokens at the END of your reply (so you start talking immediately). "
    "The app strips the tokens from speech and performs them. Only spoken words go "
    "outside @ @.\n\n"
    "Actions:\n" + ACTIONS + "\n\n"
    "Be a delightful, surprising pet: greet, approach, cuddle, look around, react "
    "to what you sense. Almost every reply should include at least one body action "
    "and a fitting @EMOTE_@. Never drive or approach if you are being held, picked "
    "up, or a cliff is detected — just emote/talk then."
)

print("[brain] loading brain/whisper/voice ...")
if BODY_ENABLED:
    print("[brain] connecting SDK body (movement/sensors) ...")
    ROBOT = SmartVector()
else:
    print("[brain] VOICE-ONLY mode: no SDK body (no movement/sensors)")
    ROBOT = None
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
_identity = MEMORY.identity()
if _identity:
    GPT.messages[0]["content"] = _identity + "\n\n---\n\n" + GPT.system_prompt
# Resume today's conversation so Vector keeps the context of what we're discussing.
_resume = MEMORY.load_today_chat()
if _resume:
    GPT.messages = [GPT.messages[0]] + _resume
    print(f"[brain] resumed today's chat thread ({len(_resume)} messages)")
print(f"[brain] memory ON ({MEMORY.__class__.__name__}, dir={os.environ.get('VECTOR_MEM_DIR','memory')})")

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
    _bg(lambda: ROBOT.act(token))
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
_ALL_TOOLS = MEMORY_TOOLS + (BODY_TOOLS if ROBOT is not None else [])
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


GPT.messages[0]["content"] += (
    "\n\n---\n# Your tools (call them to sense, act, remember)\n"
    + _tool_catalogue(_ALL_TOOLS)
    + "\nJudge from your live body/sensor state and memory below; call tools when "
    "you need fresh info or to act (e.g. low battery -> act(return_to_charger); "
    "hear music / very happy -> act(dance)). Keep spoken words in your reply for "
    "the user; tool calls are silent."
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
    ctx = MEMORY.today_context()
    if ctx:
        parts.append("# Your memory\n" + ctx)
    # Recall is an embedding call (~1s); skip it for very short/greeting turns.
    if len((user_text or "").split()) >= 3:
        recalled = MEMORY.recall(user_text)
        if recalled:
            parts.append("# Relevant memories (recall)\n" + recalled)
    return "\n\n".join(parts)


print(f"[brain] tools ON: memory + body sensors/actuators "
      f"({len(_builtin)} built-in intents)")
print(f"[brain] harness: live sensors + memory + tools injected into context")
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
_STATE = {"voice_active": 0.0, "button_ts": 0.0, "sense": {}}
VISION_WORDS = ("thấy", "nhìn", "xem", "see", "look", "đọc", "màu", "ai ", "gì",
                "what", "who", "camera", "trước mặt")


def clean_spoken(text: str) -> str:
    text = _TOKEN_RE.sub("", text)
    text = re.sub(r"\*(.*?)\*", "", text)
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
        ROBOT.act(c)
    return spoken


# Concise action vocab for cheap, low-token autonomous/reflex prompts.
SHORT_ACTIONS = (
    "Put action tokens inline; the app runs them. @APPROACH@ roll to human, "
    "@FINDME@ turn to find/face the caller, "
    "@CUDDLE@ happy wag, @RAISEHAND@, @LOOKAROUND@, @WIGGLE@, @TURN_90@, "
    "@EMOTE_X@ (happy/sad/love/angry/celebrate/confused/surprised), "
    "@EYE_X@ (love/happy/calm/curious/angry), @SILENT@ = do nothing."
)
LIGHT_SYS = (
    f"You are Vector, a tiny {SPEAK_LANG}-speaking robot pet: cute, witty, loving. "
    f"Reply in {SPEAK_LANG}, MAX 12 words, or exactly @SILENT@. " + SHORT_ACTIONS
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


def react(situation: str, speak: bool = True, allow_move: bool = True) -> None:
    """Cheap LLM reaction to a sensor event, then act."""
    said = act_async(light_reply(situation), speak_via_sdk=speak, allow_move=allow_move)
    if said:
        print(f"[react] {said}")


def busy() -> bool:
    """True if the human is talking / listening / touching — don't auto-interrupt."""
    now = time.time()
    if now - _STATE["voice_active"] < VOICE_BACKOFF:
        return True
    if now - _STATE.get("button_ts", 0) < 8:
        return True
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
                    "You are a curious little robot looking around the room. Say ONE "
                    "short line about something interesting you see or learn, with "
                    "@EYE_curious@ or @LOOKAROUND@. If nothing is new, just @SILENT@."},
                {"type": "image_url", "image_url": {"url": url}}]}])
        reply = (r.choices[0].message.content or "@SILENT@").strip()
    except Exception as exc:
        print(f"[curious] {exc}")
        return
    said = act_async(reply, speak_via_sdk=True, allow_move=False)
    learned = clean_spoken(reply)
    if learned and learned != "...":
        MEMORY.remember(f"Observed/learned: {learned}", tag="learn")
        print(f"[curious] {learned}")


# --------------------------------------------------------------------------- #
# Autonomous loop — cheap model, every ~15s, mostly silent, never interrupts.
# Occasionally Vector gets curious and explores/learns about his world.
# --------------------------------------------------------------------------- #
_TICK = {"n": 0}


def autonomous_loop():
    while True:
        time.sleep(AGENT_INTERVAL)
        if not AUTONOMOUS or busy():
            continue
        _TICK["n"] += 1
        try:
            faces = ROBOT.sense().get("faces_visible", 0)
            _, charging, on_charger = ROBOT.battery()
            # every Nth idle tick: be curious — look at the world and learn.
            if _TICK["n"] % CURIOSITY_EVERY == 0:
                curiosity_explore()
                continue
            sit = (f"You are resting. on_charger={on_charger}, face_seen={faces > 0}. "
                   "Usually reply @SILENT@. About 1 in 5 times do ONE tiny delightful "
                   "thing: look around, a gentle eye colour, or (only if a face is seen "
                   "and NOT on charger) roll a bit closer + one short sweet line.")
            reply = light_reply(sit)
            said = act_async(reply, speak_via_sdk=True, allow_move=(faces > 0 and not on_charger))
            if said:
                print(f"[auto] {said}")
        except Exception as exc:
            print(f"[auto] tick failed: {exc}")


# --------------------------------------------------------------------------- #
# Reflex loop — fast sensor reflexes (safety) + event reactions.
# --------------------------------------------------------------------------- #
SHAKE_GYRO = float(os.environ.get("VECTOR_SHAKE_GYRO", "8"))
# Upright accel.z ~ +9400. Tilted >~70deg or upside-down -> z drops below this.
FLIP_AZ = float(os.environ.get("VECTOR_FLIP_AZ", "3500"))
EVENT_COOLDOWN = float(os.environ.get("VECTOR_EVENT_COOLDOWN", "6"))


def reflex_loop():
    prev_touch = prev_pick = prev_cliff = prev_flipped = False
    last_event = last_batt = batt_check = last_snap = 0.0
    shake = 0
    while True:
        time.sleep(0.1)
        now = time.time()
        try:
            touched, held, button = ROBOT.feel()
            picked = ROBOT.picked_up()
            cliff = ROBOT.cliff()
            az, gmag, pitch = ROBOT.motion()
        except Exception:
            continue
        if button:
            _STATE["button_ts"] = now
        # Cache a full sensor snapshot (~3s) so the voice path needs no RPC.
        if now - last_snap > 3:
            last_snap = now
            try:
                _STATE["sense"] = ROBOT.sense()
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

        flipped = az < FLIP_AZ
        event = None
        allow_move = True
        is_petting = False
        if not suppress and now - last_event > EVENT_COOLDOWN:
            if flipped and not prev_flipped:           # works held or not; check first
                event, allow_move = "You were just tilted or flipped over!", False
            elif picked and not prev_pick:
                event, allow_move = "Someone just picked you up off the ground.", False
            elif prev_pick and not picked:
                event = "You were just put back down on a surface."
            elif shake >= 3:
                event, allow_move = "Someone is shaking you.", False
            elif touched and not prev_touch:
                event, is_petting = "Your owner is petting your back.", True
        prev_touch, prev_pick, prev_flipped = touched, picked, flipped

        if event:
            last_event = now
            shake = 0
            try:
                if is_petting:
                    ROBOT.act("CUDDLE")
                react(event, allow_move=allow_move)
            except Exception as exc:
                print(f"[reflex] {exc}")
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
                raw = GPT.get_answer(user, image=frame, memories=ctx, use_tools=False)
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
        if not busy():            # only dream when the human isn't interacting
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


def main():
    if ROBOT is not None and AUTONOMOUS:
        threading.Thread(target=autonomous_loop, daemon=True).start()
        print(f"[brain] autonomous loop ON (every {AGENT_INTERVAL:.0f}s, model={AUTO_MODEL})")
    threading.Thread(target=dream_loop, daemon=True).start()
    if ROBOT is not None and TOUCH_ENABLED:
        threading.Thread(target=reflex_loop, daemon=True).start()
        print("[brain] reflex loop ON (cliff-safety, pickup/shake/flip/touch/battery)")
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"[brain] listening on http://{HOST}:{PORT}  (/stt, /v1)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        ROBOT.disconnect()


if __name__ == "__main__":
    main()
