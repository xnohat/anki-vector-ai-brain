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
AGENT_INTERVAL = float(os.environ.get("VECTOR_AGENT_INTERVAL", "150"))
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
_STATE = {"voice_active": 0.0}
VISION_WORDS = ("thấy", "nhìn", "xem", "see", "look", "đọc", "màu", "ai ", "gì",
                "what", "who", "camera", "trước mặt")


def clean_spoken(text: str) -> str:
    text = _TOKEN_RE.sub("", text)
    text = re.sub(r"\*(.*?)\*", "", text)
    return re.sub(r"\s+", " ", text).strip()


def parse_commands(text: str):
    return re.findall(r"@(.*?)@", text)


def act_async(reply: str, speak_via_sdk: bool, frame_provider=None):
    """Perform the body actions in `reply`; optionally speak it via the SDK."""
    commands = parse_commands(reply)
    spoken = clean_spoken(reply)
    silent = any(c.strip().upper() == "SILENT" for c in commands)
    if ROBOT is None:
        return spoken  # voice-only: no body, no SDK speech (wire-pod speaks)

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


def sensors_summary(s: dict) -> str:
    parts = []
    for k, v in s.items():
        parts.append(f"{k}={v}")
    return ", ".join(parts)


# --------------------------------------------------------------------------- #
# Autonomous agent loop — read all sensors, let the LLM be a smart pet.
# --------------------------------------------------------------------------- #
def agent_loop():
    while True:
        time.sleep(AGENT_INTERVAL)
        if not AUTONOMOUS:
            continue
        if time.time() - _STATE["voice_active"] < VOICE_BACKOFF:
            continue
        # Don't grab control while the human is interacting (button/touch/held).
        t, h, b = ROBOT.feel()
        if t or h or b:
            continue
        try:
            s = ROBOT.sense()
            prompt = (
                "[AUTONOMOUS] No one is talking to you. Your senses right now: "
                + sensors_summary(s) + ". "
                "You are a calm robot pet resting. MOST of the time reply with EXACTLY "
                "@SILENT@ and nothing else (stay quiet, don't move). Only rarely (about "
                "1 in 4 beats) do ONE small delightful thing — a quiet look around, a "
                "gentle eye-colour change, or if a face is visible roll a little closer "
                "and say one short sweet line. Never be noisy or repetitive. If "
                "held/picked up/cliff/on charger, do NOT move — @SILENT@ or a soft word."
            )
            with _BRAIN_LOCK:
                reply = GPT.get_answer(prompt)
            # collapse the long instruction in history
            try:
                if GPT.messages[-2]["role"] == "user":
                    GPT.messages[-2]["content"] = "[autonomous sense]"
            except Exception:
                pass
            said = act_async(reply, speak_via_sdk=True, frame_provider=ROBOT.get_frame)
            if said:
                print(f"[auto] {said}")
        except Exception as exc:
            print(f"[auto] tick failed: {exc}")


# --------------------------------------------------------------------------- #
# Touch loop — backpack petting / being picked up -> cuddle + affection.
# --------------------------------------------------------------------------- #
def touch_loop():
    prev_touch = prev_held = False
    last = 0.0
    last_button = 0.0
    while True:
        time.sleep(0.05)   # poll fast so we catch the brief button click
        touched, held, button = ROBOT.feel()
        now = time.time()
        if button:
            last_button = now
        # Pressing the button also registers as a backpack touch. Suppress touch
        # reactions for a few seconds around any button press so it can't hijack
        # the "listen to me" flow with a cuddle.
        if now - last_button < 6.0:
            prev_touch, prev_held = touched, held
            continue
        event = None
        if touched and not prev_touch:
            event = "Người chủ đang vuốt ve lưng (backpack) của bạn."
        elif held and not prev_held:
            event = "Người chủ vừa bế bạn lên tay."
        prev_touch, prev_held = touched, held
        if not event:
            continue
        now = time.time()
        if now - last < TOUCH_COOLDOWN or now - _STATE["voice_active"] < VOICE_BACKOFF:
            continue
        last = now
        try:
            # snappy cuddle first, then an affectionate spoken line
            if touched:
                ROBOT.act("CUDDLE")
            with _BRAIN_LOCK:
                reply = GPT.get_answer(
                    f"[SENSOR] {event} React like a happy pet: a short loving line "
                    "in Vietnamese + a fitting action. If not held you may @CUDDLE@ "
                    "or @WIGGLE@ or @EMOTE_love@.")
            try:
                if GPT.messages[-2]["role"] == "user":
                    GPT.messages[-2]["content"] = "[touched]"
            except Exception:
                pass
            said = act_async(reply, speak_via_sdk=True)
            print(f"[touch] {said}")
        except Exception as exc:
            print(f"[touch] failed: {exc}")


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
        with _BRAIN_LOCK:
            raw = GPT.get_answer(user, image=frame)
        spoken = clean_spoken(raw) or "..."
        print(f"[chat] vector: {spoken!r}  ({[c for c in parse_commands(raw)]})")

        # Perform body actions in the background so the spoken reply isn't delayed.
        threading.Thread(target=_deferred_act, args=(raw,), daemon=True).start()

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


def main():
    if ROBOT is not None and AUTONOMOUS:
        threading.Thread(target=agent_loop, daemon=True).start()
        print(f"[brain] autonomous pet loop ON (every {AGENT_INTERVAL:.0f}s)")
    if ROBOT is not None and TOUCH_ENABLED:
        threading.Thread(target=touch_loop, daemon=True).start()
        print("[brain] touch/backpack reactions ON (button press ignored = listen)")
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"[brain] listening on http://{HOST}:{PORT}  (/stt, /v1)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        ROBOT.disconnect()


if __name__ == "__main__":
    main()
