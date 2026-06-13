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
    "@CUDDLE@ happy-dog wiggle with your hand(lift) and wheels. "
    "@RAISEHAND@ raise your lift like raising a hand. @LOWERHAND@ lower it. "
    "@LOOKAROUND@ glance around the room. @WIGGLE@ a playful wiggle. "
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
    "not just words. Put @COMMAND@ action tokens inline in your reply; the app "
    "strips them from speech and performs them. Only spoken words go outside @ @.\n\n"
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
print(f"[brain] memory ON ({MEMORY.__class__.__name__}, dir={os.environ.get('VECTOR_MEM_DIR','memory')})")
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
_STATE = {"voice_active": 0.0, "button_ts": 0.0}
VISION_WORDS = ("thấy", "nhìn", "xem", "see", "look", "đọc", "màu", "ai ", "gì",
                "what", "who", "camera", "trước mặt")


def clean_spoken(text: str) -> str:
    text = _TOKEN_RE.sub("", text)
    text = re.sub(r"\*(.*?)\*", "", text)
    return re.sub(r"\s+", " ", text).strip()


def parse_commands(text: str):
    return re.findall(r"@(.*?)@", text)


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
            messages=[{"role": "user", "content": [
                {"type": "text", "text":
                    "Bạn là Vector, robot nhỏ tò mò đang quan sát căn phòng. Nói MỘT câu "
                    "ngắn tiếng Việt về một điều thú vị bạn thấy hoặc học được, kèm "
                    "@EYE_curious@ hoặc @LOOKAROUND@. Nếu không có gì mới, chỉ @SILENT@."},
                {"type": "image_url", "image_url": {"url": url}}]}])
        reply = (r.choices[0].message.content or "@SILENT@").strip()
    except Exception as exc:
        print(f"[curious] {exc}")
        return
    said = act_async(reply, speak_via_sdk=True, allow_move=False)
    learned = clean_spoken(reply)
    if learned and learned != "...":
        MEMORY.remember(f"Quan sát/học được: {learned}", tag="learn")
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
    prev_touch = prev_pick = prev_cliff = False
    last_event = last_batt = batt_check = 0.0
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

        # ---- SAFETY (instant, no LLM): table edge -> stop the wheels NOW ----
        if cliff and not prev_cliff:
            ROBOT.stop()
            print("[reflex] CLIFF! wheels stopped")
        prev_cliff = cliff

        shake = min(6, shake + 1) if gmag > SHAKE_GYRO else max(0, shake - 1)
        suppress = busy() or (now - _STATE.get("button_ts", 0) < 6)

        event = None
        allow_move = True
        if not suppress and now - last_event > EVENT_COOLDOWN:
            if picked and not prev_pick:
                event, allow_move = "Người chủ vừa nhấc bạn lên khỏi mặt đất.", False
            elif prev_pick and not picked:
                event = "Bạn vừa được đặt xuống mặt phẳng."
            elif az < FLIP_AZ and not picked:
                event, allow_move = "Bạn vừa bị lật nghiêng hoặc úp ngược.", False
            elif shake >= 3:
                event, allow_move = "Ai đó đang lắc người bạn.", False
            elif touched and not prev_touch:
                event = "Người chủ đang vuốt ve lưng bạn."
        prev_touch, prev_pick = touched, picked

        if event:
            last_event = now
            shake = 0
            try:
                if "vuốt" in event:
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
                    react("Pin sắp cạn! Kêu cứu thật đáng yêu bằng tiếng Việt.")
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
        # QMD recall: pull relevant long-term memories into this answer.
        mems = MEMORY.recall(user)
        with _BRAIN_LOCK:
            raw = GPT.get_answer(user, image=frame, memories=mems)
        spoken = clean_spoken(raw) or "..."
        print(f"[chat] vector: {spoken!r}  ({[c for c in parse_commands(raw)]})")

        # Perform body actions in the background so the spoken reply isn't delayed.
        threading.Thread(target=_deferred_act, args=(raw,), daemon=True).start()
        # Remember the exchange + learn about new people, in the background.
        threading.Thread(target=_post_chat, args=(user, spoken), daemon=True).start()

        if stream:
            self._stream(spoken)
        else:
            self._json(200, {"id": "chatcmpl-vb", "object": "chat.completion",
                             "created": int(time.time()), "model": GPT.model,
                             "choices": [{"index": 0, "finish_reason": "stop",
                                          "message": {"role": "assistant", "content": spoken}}]})

    def _stream(self, text):
        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            base = {"id": "chatcmpl-vb", "object": "chat.completion.chunk",
                    "created": int(time.time()), "model": GPT.model}

            def send(delta, finish=None):
                ch = dict(base); ch["choices"] = [{"index": 0, "delta": delta, "finish_reason": finish}]
                self.wfile.write(f"data: {json.dumps(ch)}\n\n".encode()); self.wfile.flush()

            send({"role": "assistant"})
            for w in text.split(" "):
                send({"content": w + " "})
            send({}, "stop")
            self.wfile.write(b"data: [DONE]\n\n"); self.wfile.flush()
        except (BrokenPipeError, ConnectionError):
            pass
        except Exception as exc:
            print(f"[chat] stream err: {exc}")


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
                {"type": "text", "text": "Mô tả ngắn người trong ảnh (vẻ ngoài, đặc "
                 "điểm dễ nhớ) bằng tiếng Việt, 1-2 câu."},
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
    MEMORY.save_user(name, desc or "(chưa rõ vẻ ngoài)", pic_jpeg=pic)
    MEMORY.remember(f"Lần đầu gặp {name}." + (f" {desc}" if desc else ""), tag="people")
    print(f"[memory] met new person: {name}")


def _post_chat(user: str, spoken: str) -> None:
    """After a voice exchange: write it to the journal + learn about new people."""
    try:
        MEMORY.remember(f"Người dùng nói \"{user}\". Vector đáp \"{spoken}\".", tag="chat")
        _maybe_meet_person(user)
    except Exception as exc:
        print(f"[memory] post_chat failed: {exc}")


def consolidate_loop():
    """Vector Brain consolidates the journals into long-term MEMORY.md daily."""
    interval = float(os.environ.get("VECTOR_CONSOLIDATE_HOURS", "8")) * 3600
    while True:
        time.sleep(interval)
        try:
            if MEMORY.consolidate():
                print("[memory] consolidated journals -> MEMORY.md")
        except Exception as exc:
            print(f"[memory] consolidate loop: {exc}")


def main():
    if ROBOT is not None and AUTONOMOUS:
        threading.Thread(target=autonomous_loop, daemon=True).start()
        print(f"[brain] autonomous loop ON (every {AGENT_INTERVAL:.0f}s, model={AUTO_MODEL})")
    threading.Thread(target=consolidate_loop, daemon=True).start()
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
