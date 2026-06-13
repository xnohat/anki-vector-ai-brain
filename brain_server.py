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
AGENT_INTERVAL = float(os.environ.get("VECTOR_AGENT_INTERVAL", "75"))
TOUCH_COOLDOWN = float(os.environ.get("VECTOR_TOUCH_COOLDOWN", "8"))
VOICE_BACKOFF = float(os.environ.get("VECTOR_VOICE_BACKOFF", "10"))
AUTONOMOUS = os.environ.get("VECTOR_AUTONOMOUS", "1") not in ("0", "false", "False", "")

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

print("[brain] connecting to Vector + loading brain/whisper/voice ...")
ROBOT = SmartVector()
GPT = CustomGPT(system_prompt=UNIFIED_PROMPT)
STT = WhisperSTT()
VOICE = Voice()
print(f"[brain] ready: model={GPT.model}, stt={STT.language or 'auto'}")

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
        try:
            s = ROBOT.sense()
            prompt = (
                "[AUTONOMOUS] No one is talking to you. Your senses right now: "
                + sensors_summary(s) + ". "
                "Decide what to do as a living robot pet. Mostly be calm, but every "
                "so often do something delightful and unexpected — look around, roll "
                "closer to your human and say you missed them, raise your hand, "
                "wiggle, change eye colour. If a face is visible, you may @APPROACH@. "
                "If held/picked up/cliff, do NOT move — just emote or speak softly. "
                "Reply with a short line + action tokens, or exactly @SILENT@."
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
    while True:
        time.sleep(0.25)
        touched, held = ROBOT.feel()
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
            text = (STT.inference(path) or "").strip()
            os.unlink(path)
            print(f"[stt] -> {text!r}")
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
        if any(w in low for w in VISION_WORDS):
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
    # tiny delay so wire-pod's speech leads, then act the body
    time.sleep(0.4)
    try:
        act_async(raw, speak_via_sdk=False, frame_provider=ROBOT.get_frame)
    except Exception as exc:
        print(f"[act] {exc}")
    _STATE["voice_active"] = time.time()


def main():
    if AUTONOMOUS:
        threading.Thread(target=agent_loop, daemon=True).start()
        print(f"[brain] autonomous pet loop ON (every {AGENT_INTERVAL:.0f}s)")
    threading.Thread(target=touch_loop, daemon=True).start()
    print("[brain] touch/backpack reactions ON")
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"[brain] listening on http://{HOST}:{PORT}  (/stt, /v1)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        ROBOT.disconnect()


if __name__ == "__main__":
    main()
