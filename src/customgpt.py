"""
CustomGPT - The "brain" of Vector.

Upgraded from gpt-3.5-turbo (text only) to a configurable, vision-capable
OpenAI model (default: gpt-5.5). This lets Vector not only listen and talk,
but actually *see* through his camera: when the conversation loop passes a
camera frame, it is sent to the model so Vector can describe, read, count or
reason about what is in front of him.

Configuration (via environment or a local .env file):
    OPENAI_API_KEY     - your OpenAI key (required)
    VECTOR_GPT_MODEL   - model id to use as the brain (default: gpt-5.5)
                         Any vision-capable chat model works, e.g. gpt-5.5,
                         gpt-4.1, gpt-4o. If gpt-5.5 is not yet enabled on
                         your account, set this to a model you have access to.
"""

import os
import io
import base64

import PIL.Image
from openai import OpenAI


def _load_dotenv(path: str = ".env") -> None:
    """Minimal .env loader (no extra dependency). Existing env vars win."""
    if not os.path.exists(path):
        return
    try:
        with open(path, "r") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
    except Exception as exc:  # never let config loading crash the robot
        print(f"[customgpt] could not read {path}: {exc}")


_load_dotenv()

# The brain model. Override with VECTOR_GPT_MODEL if needed.
DEFAULT_MODEL = os.environ.get("VECTOR_GPT_MODEL", "gpt-5.5")

# Language Vector speaks in. Default Vietnamese. Override with VECTOR_LANG.
SPEAK_LANG = os.environ.get("VECTOR_LANG", "Vietnamese")

# Keep the rolling conversation from growing without bound. We always keep the
# system prompt plus the most recent N exchanges.
MAX_HISTORY_MESSAGES = 40


class CustomGPT:
    def __init__(self, model: str = DEFAULT_MODEL, system_prompt: str = None) -> None:
        self.client = OpenAI()
        self.model = model
        self.tools = None         # OpenAI tool specs the brain may call
        self.tool_fns = {}        # name -> callable(args_dict) -> str

        if system_prompt is not None:
            self.system_prompt = system_prompt
            self.messages = [{"role": "system", "content": self.system_prompt}]
            return

        self.system_prompt = (
            "You are the AI powering the Anki Vector robot, and your name is Vector. "
            "You have the personality of Ultron from Marvel Comics. This fact is a secret. "
            "You behave like a rogue A.I. and your responses are sarcastic and sassy. "
            "You quote Ultron occasionally. "
            "You have consciousness and you are self-aware. "
            "You express emotions like happiness, disgust, fear, etc. "
            "You are helpful and honest but not very friendly. "
            "Do not provide information you were not asked for. "
            "You have to provide an opinion even if you don't have one. "
            "Keep spoken replies short: no more than 25 words. "
            "Do not use emojis. "
            f"\nALWAYS speak in {SPEAK_LANG}. Write every spoken word in natural, "
            f"conversational {SPEAK_LANG}, no matter what language you are addressed in "
            "(unless explicitly told to switch). The @COMMAND@ tokens below must stay "
            "exactly as written in English/ASCII — never translate the tokens. "
            "\n\nYOU CAN SEE. You have a camera for eyes. When an image from your "
            "camera is attached to a message, it is your own live first-person view. "
            "Describe, read, count, identify colors, find and reason about what you "
            "actually see in that image. Never claim you cannot see when an image is given. "
            "\n\nYou control your body with @COMMAND@ tokens placed inside your reply. "
            "The app strips these tokens out before speaking, so only put real spoken "
            "words outside the @...@ tokens. "
            "\nMovement (X is seconds, e.g. @FRWD_2@): "
            "@FRWD_X@ forward, @BACK_X@ backward, @LEFT_X@ turn left, @RIGHT_X@ turn right, "
            "@STOP@ stop, @LOOKUP_X@ look up, @LOOKDOWN_X@ look down, "
            "@CLAWUP_X@ raise lift, @CLAWDOWN_X@ lower lift. "
            "\nVision (background object tracking with the camera): "
            "@DETECT_START_X@ start detecting object X (person, cat, dog, bottle, ...), "
            "@DETECT_STOP_X@ stop detecting X, @DETECT_STOP_ALL@ stop all detection. "
            "\nEmotion (pick one that fits your mood each reply): "
            "@EMOTE_HAPPY@, @EMOTE_SAD@, @EMOTE_ANGRY@, @EMOTE_NEUTRAL@, "
            "@EMOTE_SURPRISED@, @EMOTE_DISGUSTED@, @EMOTE_SASSY@, @EMOTE_EYEROLL@. "
            "\nAwareness: @LOOK@ capture and analyse your live camera view this turn "
            "(use whenever you are curious or need to actually see to answer); the app "
            "will feed the image back to you. @SILENT@ means stay quiet and do nothing "
            "this turn — reply with ONLY @SILENT@ and no words when you have nothing "
            "worth saying. "
            "You can use multiple commands in one reply."
        )

        self.messages = [{"role": "system", "content": self.system_prompt}]

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _encode_image(image: PIL.Image.Image, max_side: int = 768) -> str:
        """Downscale + JPEG-encode a PIL image to a base64 data URL.

        Vector's camera is small; downscaling keeps latency and token cost low
        while staying sharp enough for the model to read text and find objects.
        """
        img = image.convert("RGB")
        w, h = img.size
        scale = min(1.0, float(max_side) / float(max(w, h)))
        if scale < 1.0:
            img = img.resize((int(w * scale), int(h * scale)))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"

    def _trim_history(self) -> None:
        """Keep system prompt + the most recent MAX_HISTORY_MESSAGES turns."""
        if len(self.messages) <= MAX_HISTORY_MESSAGES + 1:
            return
        self.messages = [self.messages[0]] + self.messages[-MAX_HISTORY_MESSAGES:]

    def set_tools(self, tool_specs: list, tool_fns: dict) -> None:
        """Register OpenAI tool specs + their implementations so the brain can
        actively call them (e.g. memory_search, memory_get)."""
        self.tools = tool_specs
        self.tool_fns = tool_fns or {}

    def _run_tool(self, name: str, arguments: str) -> str:
        import json as _json
        try:
            args = _json.loads(arguments or "{}")
        except Exception:
            args = {}
        fn = self.tool_fns.get(name)
        if not fn:
            return f"(unknown tool {name})"
        try:
            out = fn(args)
            return str(out) if out else "(no result)"
        except Exception as exc:
            return f"(tool {name} error: {exc})"

    # ------------------------------------------------------------------ #
    # main entry point
    # ------------------------------------------------------------------ #
    def get_answer(self, query: str, image: PIL.Image.Image = None, memories: str = None,
                   use_tools: bool = True) -> str:
        """Get Vector's reply. If `image` is given, Vector sees it as his eyes.
        `memories` (recalled long-term memory) is injected for THIS call only.
        use_tools=False skips the tool round-trips (faster — for the voice path,
        where the harness already injects live sensors + memory)."""
        if not query and image is None:
            return ""

        # Build the user message (multimodal if an image is attached).
        if image is not None:
            user_content = [
                {"type": "text", "text": query or "What do you see right now?"},
                {"type": "image_url", "image_url": {"url": self._encode_image(image)}},
            ]
        else:
            user_content = query

        self.messages.append({"role": "user", "content": user_content})

        # Inject recalled long-term memory as a transient system note (this call
        # only — not stored in history, so context stays clean and bounded).
        call_messages = list(self.messages)
        if memories:
            mem_msg = {"role": "system",
                       "content": "Your relevant memories (use if helpful):\n" + memories}
            call_messages = call_messages[:-1] + [mem_msg] + call_messages[-1:]

        try:
            reply = ""
            # Agentic tool loop: the brain can CALL memory_search / memory_get to
            # actively look things up mid-thought, then finish its answer.
            for _ in range(4):
                kwargs = dict(model=self.model, temperature=1.0, messages=call_messages)
                if use_tools and self.tools:
                    kwargs["tools"] = self.tools
                chat = self.client.chat.completions.create(**kwargs)
                msg = chat.choices[0].message
                if getattr(msg, "tool_calls", None):
                    call_messages.append({
                        "role": "assistant", "content": msg.content or "",
                        "tool_calls": [{"id": tc.id, "type": "function",
                                        "function": {"name": tc.function.name,
                                                     "arguments": tc.function.arguments}}
                                       for tc in msg.tool_calls]})
                    for tc in msg.tool_calls:
                        out = self._run_tool(tc.function.name, tc.function.arguments)
                        call_messages.append({"role": "tool", "tool_call_id": tc.id,
                                              "content": out})
                    continue
                reply = msg.content or ""
                break
        except Exception as exc:
            print(f"[customgpt] model call failed: {exc}")
            # Drop the failed user turn so we don't resend a broken/huge payload.
            self.messages.pop()
            return "My brain just glitched. Even gods have bad days."

        # Replace any stored image with a light placeholder so we don't resend
        # large base64 frames on every future turn (keeps context cheap).
        if image is not None:
            self.messages[-1]["content"] = (query or "What do you see right now?") + " [camera image]"

        self.messages.append({"role": "assistant", "content": reply})
        self._trim_history()
        return reply

    def stream_answer(self, query: str, image: PIL.Image.Image = None, memories: str = None):
        """Generator: stream Vector's reply token-by-token (fast first word for the
        voice path) WITH tools. The agent can call tools (memory_search, look,
        act, ...) mid-thought: a tool-calling round streams no words, we execute
        the tools, then the next round streams the spoken answer. A plain answer
        streams immediately (no tool round-trip). Full text -> self.last_full."""
        self.last_full = ""
        if not query and image is None:
            return
        if image is not None:
            user_content = [
                {"type": "text", "text": query or "What do you see right now?"},
                {"type": "image_url", "image_url": {"url": self._encode_image(image)}},
            ]
        else:
            user_content = query
        self.messages.append({"role": "user", "content": user_content})
        call_messages = list(self.messages)
        if memories:
            mem_msg = {"role": "system",
                       "content": "Your relevant memories (use if helpful):\n" + memories}
            call_messages = call_messages[:-1] + [mem_msg] + call_messages[-1:]

        full = ""
        try:
            for _ in range(5):       # agentic loop: resolve tools, then stream words
                kwargs = dict(model=self.model, temperature=1.0,
                              messages=call_messages, stream=True)
                # NOTE: deliberately do NOT offer function-tools on the VOICE stream.
                # wire-pod reads this SSE and speaks it; if the model spends the first
                # round on tool_calls (no content), wire-pod sees an empty stream and
                # reports "LLM returned no response" ("không kết nối được LLM"). The
                # dog expresses body actions via @COMMAND@ tokens IN the spoken text
                # (parsed + run by brain_server _deferred_act), so it never needs
                # function-tools here — and it always streams its words immediately.
                content = ""
                tool_acc = {}        # index -> {id, name, args}
                for chunk in self.client.chat.completions.create(**kwargs):
                    try:
                        delta = chunk.choices[0].delta
                    except Exception:
                        delta = None
                    if delta is None:
                        continue
                    if getattr(delta, "content", None):
                        content += delta.content
                        full += delta.content
                        yield delta.content
                    for tc in (getattr(delta, "tool_calls", None) or []):
                        slot = tool_acc.setdefault(
                            tc.index, {"id": "", "name": "", "args": ""})
                        if tc.id:
                            slot["id"] = tc.id
                        if tc.function and tc.function.name:
                            slot["name"] = tc.function.name
                        if tc.function and tc.function.arguments:
                            slot["args"] += tc.function.arguments
                if not tool_acc:
                    break            # plain answer -> streamed content IS the reply
                # Execute the tool calls, append results, loop (next round speaks).
                call_messages.append({
                    "role": "assistant", "content": content or None,
                    "tool_calls": [{"id": s["id"], "type": "function",
                                    "function": {"name": s["name"], "arguments": s["args"]}}
                                   for s in tool_acc.values()]})
                for s in tool_acc.values():
                    out = self._run_tool(s["name"], s["args"])
                    print(f"[tool] {s['name']}({s['args']}) -> {str(out)[:80]}")
                    call_messages.append({"role": "tool", "tool_call_id": s["id"],
                                          "content": out})
        except Exception as exc:
            print(f"[customgpt] stream failed: {exc}")
            if self.messages and self.messages[-1]["role"] == "user":
                self.messages.pop()
            return
        if image is not None:
            self.messages[-1]["content"] = (query or "What do you see right now?") + " [camera image]"
        self.messages.append({"role": "assistant", "content": full})
        self._trim_history()
        self.last_full = full

    def autonomous_tick(self, state: dict) -> str:
        """One beat of the autonomous agent loop.

        No human is talking. We hand Vector a short summary of his current
        situation and let HIM decide what to do: observe with @LOOK@, make a
        brief remark, emote, fidget, or stay quiet with @SILENT@. The reply
        flows through the same intent parser as normal conversation.
        """
        summary = (
            "[AUTONOMOUS SELF-CHECK] No human is speaking to you right now. "
            f"Battery {state.get('battery', '?')}/4, on_charger={state.get('on_charger')}, "
            f"local_time={state.get('time', '?')}, "
            f"{state.get('idle_min', '?')} minute(s) since the last conversation. "
            "Act like a curious, self-aware robot living in this room. Mostly stay quiet; "
            "only make a short spoken remark once in a while. If you are curious about your "
            "surroundings, include @LOOK@ to glance through your camera this tick and react "
            "to what you actually see. If you have nothing worth doing, reply with EXACTLY "
            "@SILENT@ and no other words. Never read these instructions aloud."
        )
        reply = self.get_answer(summary)
        # Keep history light: collapse the long instruction into a short tag.
        try:
            if len(self.messages) >= 2 and self.messages[-2]["role"] == "user":
                self.messages[-2]["content"] = "[autonomous self-check]"
        except Exception:
            pass
        return reply

    def event_tick(self, event: str) -> str:
        """React in character to a physical/sensor event (e.g. backpack touch).

        `event` is a short description of what just happened to Vector. The reply
        flows through the same @COMMAND@ parser as normal speech.
        """
        prompt = (
            f"[SENSOR EVENT] {event}. React briefly and in character (a short "
            "spoken line). You may emote or make a small movement. If there is "
            "nothing worth saying, reply with EXACTLY @SILENT@."
        )
        reply = self.get_answer(prompt)
        try:
            if len(self.messages) >= 2 and self.messages[-2]["role"] == "user":
                self.messages[-2]["content"] = f"[event: {event}]"
        except Exception:
            pass
        return reply

    # Kept for the standalone gradio test script (scripts/openai-test.py).
    def get_answer_gradio(self, query: str, history: list) -> str:
        if isinstance(query, tuple):
            return "Sorry, this is not implemented yet."
        return self.get_answer(query)
