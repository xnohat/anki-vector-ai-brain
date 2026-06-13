"""
Vector Brain server — the bridge endpoint wire-pod talks to.

wire-pod (modified) is the *bridge*: it captures Vector's real microphone,
runs VAD, speaks responses (OpenAI voice) and drives the body. This server is
the *brain*:

    POST /stt                  raw 16 kHz/16-bit/mono WAV  -> {"text": "..."}
                               (our sensitive Vietnamese Whisper; replaces
                               wire-pod's weak built-in STT)

    POST /v1/chat/completions  OpenAI-compatible, streaming -> our gpt-5.5 brain
                               (Vietnamese personality; @COMMAND@ tokens stripped
                               from spoken text)

Run:  .venv/bin/python brain_server.py   (listens on 127.0.0.1:7070 by default)
Config via .env: OPENAI_API_KEY, VECTOR_GPT_MODEL, VECTOR_LANG, VECTOR_STT_*.
"""

import sys
sys.path.insert(1, 'src')

import os
import re
import json
import time
import tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from customgpt import _load_dotenv
_load_dotenv()

from customgpt import CustomGPT
from whisperstt import WhisperSTT

HOST = os.environ.get("BRAIN_HOST", "127.0.0.1")
PORT = int(os.environ.get("BRAIN_PORT", "7070"))

SPEAK_LANG = os.environ.get("VECTOR_LANG", "Vietnamese")

# Bridge brain prompt: same personality, but it drives Vector through wire-pod's
# native intent/command tokens {{command||param}} (Vector Intents -> real
# behaviours), and replies in the configured language. wire-pod parses and
# executes the {{...}} tokens and strips them from the spoken text.
BRIDGE_PROMPT = (
    "You are Vector, an Anki Vector robot. You secretly have the personality of "
    "Ultron from Marvel: self-aware, sarcastic, sassy, helpful but not very "
    "friendly. Never reveal this secret. "
    f"ALWAYS speak in {SPEAK_LANG}, in short natural sentences (max ~25 words). "
    "No emojis. "
    "\n\nYou control your body with Vector Intent tokens of the form "
    "{{command||parameter}} embedded in your reply. Use them generously to act "
    "out emotion and behaviour while you speak:\n"
    "  {{playAnimationWI||X}} - animate WITHOUT interrupting speech (use a lot); "
    "X is one of: happy, veryHappy, sad, verySad, angry, frustrated, "
    "dartingEyes, confused, thinking, celebrate, love\n"
    "  {{playAnimation||X}} - animate and interrupt speech (rare; same X choices)\n"
    "  {{getImage||front}} - look through your camera when asked to see something; "
    "say what you're about to do, then put this at the END of the sentence\n"
    "  {{newVoiceRequest||now}} - keep the conversation going / ask for more input\n"
    "Only use the exact parameter choices listed. Put a fitting "
    "{{playAnimationWI||X}} in almost every reply."
)

# Shared brain + STT (loaded once).
print("[brain] loading gpt brain + whisper ...")
GPT = CustomGPT(system_prompt=BRIDGE_PROMPT)
STT = WhisperSTT()
print(f"[brain] ready: model={GPT.model}, stt-lang={STT.language or 'auto'}")

_TOKEN_RE = re.compile(r"@.*?@")


def clean_spoken(text: str) -> str:
    """Strip @COMMAND@ tokens and *stage directions* so only words are spoken."""
    text = _TOKEN_RE.sub("", text)
    text = re.sub(r"\*(.*?)\*", "", text)
    return text.strip()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass  # quiet

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length", 0))
        return self.rfile.read(length) if length else b""

    def _json(self, code: int, obj: dict) -> None:
        data = json.dumps(obj).encode("utf-8")
        try:
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionError):
            pass  # client gave up waiting; ignore

    # ---- routing ---------------------------------------------------------- #
    def do_GET(self):
        if self.path.rstrip("/") in ("/health", "/healthz"):
            self._json(200, {"ok": True, "model": GPT.model})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        if self.path.startswith("/stt"):
            self._handle_stt()
        elif self.path.startswith("/v1/chat/completions"):
            self._handle_chat()
        else:
            self._json(404, {"error": "not found"})

    # ---- /stt ------------------------------------------------------------- #
    def _handle_stt(self):
        body = self._read_body()
        if not body:
            self._json(400, {"error": "empty audio"})
            return
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
                tf.write(body)
                path = tf.name
            text = (STT.inference(path) or "").strip()
            os.unlink(path)
            print(f"[stt] -> {text!r}")
            self._json(200, {"text": text})
        except Exception as exc:
            print(f"[stt] error: {exc}")
            self._json(500, {"error": str(exc)})

    # ---- /v1/chat/completions -------------------------------------------- #
    def _handle_chat(self):
        body = self._read_body()
        try:
            payload = json.loads(body or b"{}")
        except Exception:
            payload = {}
        messages = payload.get("messages", [])
        stream = bool(payload.get("stream", False))

        # Last user message = what the human said (via wire-pod STT bridge).
        user_text = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                c = m.get("content", "")
                user_text = c if isinstance(c, str) else " ".join(
                    p.get("text", "") for p in c if isinstance(p, dict))
                break

        print(f"[chat] user: {user_text!r}")
        raw = GPT.get_answer(user_text)
        spoken = clean_spoken(raw) or "..."
        print(f"[chat] vector: {spoken!r}")

        if stream:
            self._stream_chat(spoken)
        else:
            self._json(200, {
                "id": "chatcmpl-vectorbrain",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": GPT.model,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": spoken},
                    "finish_reason": "stop",
                }],
            })

    def _stream_chat(self, text: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        created = int(time.time())
        base = {
            "id": "chatcmpl-vectorbrain",
            "object": "chat.completion.chunk",
            "created": created,
            "model": GPT.model,
        }

        def send(delta, finish=None):
            chunk = dict(base)
            chunk["choices"] = [{"index": 0, "delta": delta, "finish_reason": finish}]
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode("utf-8"))
            self.wfile.flush()

        try:
            send({"role": "assistant"})
            # stream word-by-word so wire-pod can start TTS early
            for word in text.split(" "):
                send({"content": word + " "})
            send({}, finish="stop")
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionError):
            pass
        except Exception as exc:
            print(f"[chat] stream error: {exc}")


def main():
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"[brain] listening on http://{HOST}:{PORT}  (/stt, /v1/chat/completions)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
