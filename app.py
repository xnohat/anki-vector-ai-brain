import sys
sys.path.insert(1, 'src')
from typing import Tuple, Optional
import os
import time
import difflib
from threading import Thread, Lock

# Source
from ui import UserInterface
from vectorbot import VectorBot, Data, Action
from speechstream import StreamHandler
from customgpt import CustomGPT
from voice import Voice

# --------------------------------------------------------------------------- #
# Configuration (override via .env / environment)
# --------------------------------------------------------------------------- #
# Seconds between autonomous agent-loop ticks.
AGENT_INTERVAL = float(os.environ.get("VECTOR_AGENT_INTERVAL", "90"))
# After the wake word, stay "active" for this long so follow-ups need no wake word.
CONVO_WINDOW = float(os.environ.get("VECTOR_CONVO_WINDOW", "25"))
# Turn the autonomous loop on/off.
AUTONOMOUS = os.environ.get("VECTOR_AUTONOMOUS", "1") not in ("0", "false", "False", "")
# React to backpack touch / being picked up.
SENSORS = os.environ.get("VECTOR_SENSORS", "1") not in ("0", "false", "False", "")
# Min seconds between touch-triggered reactions (avoid spamming the brain).
TOUCH_COOLDOWN = float(os.environ.get("VECTOR_TOUCH_COOLDOWN", "8"))

# Wake words. Whisper runs in English mode, so a Vietnamese pronunciation of
# "vector" (véc-tơ) is transcribed phonetically — we match several spellings
# plus a fuzzy fallback.
WAKE_WORDS = [
    "vector", "victor", "vecto", "vektor", "vec to", "vec tor", "vichto",
    "vich to", "veto", "wecto", "hey vector", "ok vector", "vegeta", "vic to",
    "véc tơ", "véc to", "vít to", "vít tơ",
]
_custom_wake = os.environ.get("VECTOR_WAKE_WORDS", "").strip()
if _custom_wake:
    WAKE_WORDS = [w.strip().lower() for w in _custom_wake.split(",") if w.strip()]


def detect_wake_word(text: str) -> bool:
    """True if the wake word appears (substring or fuzzy match on early words)."""
    low = " " + text.lower().strip() + " "
    for w in WAKE_WORDS:
        if w in low:
            return True
    # Fuzzy match against the first couple of spoken words (handles accents/noise).
    words = text.lower().split()
    for token in words[:3]:
        token = "".join(ch for ch in token if ch.isalpha())
        if not token:
            continue
        for w in WAKE_WORDS:
            if " " in w:
                continue
            if difflib.SequenceMatcher(None, token, w).ratio() >= 0.8:
                return True
    return False


# Words that mean "use your eyes" -> attach a live camera frame to the brain.
VISION_TRIGGERS = (
    "see", "look", "watch", "show", "describe", "read", "colour", "color",
    "find", "recogni", "identif", "count", "holding", "in front", "camera",
    "your eye", "view", "what is this", "what's this", "who is", "whats this",
    "what do you", "can you see", "scan",
)


def wants_vision(text: str) -> bool:
    """Heuristic: does this utterance explicitly ask Vector to use his camera?"""
    low = text.lower()
    return any(trigger in low for trigger in VISION_TRIGGERS)


def parse_commands(text: str) -> Tuple[str, list]:
    import re
    # Remove all \n and \t
    text = text.replace("\n", "").replace("\t", "")
    # Replace AI with A.I.
    text = text.replace("AI", "A.I.")
    # Remove all text between * and *
    remove = re.sub(r'\*(.*?)\*', '', text)
    for r in remove:
        text = text.replace(f"*{r}*", "")
    # Extract all @command@ tokens
    commands = re.findall(r'@(.*?)@', text)
    for command in commands:
        text = text.replace(f"@{command}@", "")
    return text.strip(), commands


# --------------------------------------------------------------------------- #
# Shared mutable state across the two worker threads
# --------------------------------------------------------------------------- #
class Brain:
    def __init__(self) -> None:
        self.lock = Lock()                 # serialize robot/GPT use across threads
        self.convo_active_until = 0.0      # wake-word conversation window
        self.last_interaction = time.time()


def speak(text: str, robot_action: Action, voice: Voice, handler: StreamHandler) -> None:
    """Speak `text` with Vietnamese OpenAI voice (falls back to onboard TTS).

    Mutes the mic during playback so Vector doesn't transcribe his own voice.
    """
    if not text:
        return
    if handler is not None:
        try:
            handler.asst.talking = True   # tell the mic callback to ignore input
        except Exception:
            pass
    try:
        wav = voice.synthesize(text) if voice else None
        if not (wav and robot_action.say_wav(wav)):
            robot_action.tts(text)    # fallback: onboard engine
    finally:
        if handler is not None:
            try:
                handler.asst.talking = False
            except Exception:
                pass


def run_turn(
    user_input: str,
    want_frame: bool,
    gpt: CustomGPT,
    robot_data: Data,
    robot_action: Action,
    ui: UserInterface,
    voice: Voice,
    handler: StreamHandler,
    speaker_label: str = "Vector",
) -> None:
    """Run one GPT turn end to end: think -> (optionally look) -> speak -> act.

    Supports LLM-triggered vision: if the model replies with @LOOK@ and we did
    not already attach a frame, we grab one and ask it again with the image.
    """
    frame = robot_data.get_pil_frame() if want_frame else None

    reply = gpt.get_answer(user_input, image=frame)
    _, commands = parse_commands(reply)

    # The model can decide on its own that it needs to see.
    if frame is None and any(c == "LOOK" for c in commands):
        robot_action.emote("TakeAPictureFocusing")
        frame = robot_data.get_pil_frame()
        if frame is not None:
            reply = gpt.get_answer(
                "Here is your live camera view. React to what you actually see.",
                image=frame,
            )

    robot_action.emote("KnowledgeGraphSearchingGetOutSuccess")
    text, commands = parse_commands(reply)

    silent = any(c == "SILENT" for c in commands)
    # LOOK/SILENT are control tokens, not body commands -> never hand to motors.
    commands = [c for c in commands if c not in ("LOOK", "SILENT")]

    if text and not silent:
        speak(text, robot_action, voice, handler)
        ui.add_text(speaker_label, text)
    robot_action.emote("NeutralFace")
    if commands:
        robot_action.manage_commands(commands)


def conversation(
        ui: UserInterface,
        handler: StreamHandler,
        gpt: CustomGPT,
        robot_data: Data,
        robot_action: Action,
        brain: Brain,
        voice: Voice,
    ) -> None:
    """Active interaction loop: only engages the brain on wake word / while active."""
    while True:
        # Listening emote feedback while the user is speaking.
        if handler.speaking:
            robot_action.emote('OnboardingWakeWordGetIn')

        if not isinstance(handler.stt_result, type(None)):
            user_input = handler.stt_result
            handler.stt_result = None

            now = time.time()
            active = now < brain.convo_active_until
            woke = detect_wake_word(user_input)

            if woke or active:
                with brain.lock:
                    brain.convo_active_until = time.time() + CONVO_WINDOW
                    ui.add_text("Me", user_input)
                    robot_action.emote('KnowledgeGraphListening')
                    run_turn(
                        user_input,
                        want_frame=wants_vision(user_input),
                        gpt=gpt,
                        robot_data=robot_data,
                        robot_action=robot_action,
                        ui=ui,
                        voice=voice,
                        handler=handler,
                    )
                    brain.last_interaction = time.time()
                    brain.convo_active_until = time.time() + CONVO_WINDOW
            else:
                # Heard speech but not addressed to Vector -> ignore (no API cost).
                print(f"[idle] ignored (no wake word): {user_input!r}")

        time.sleep(0.25)


def gather_state(vector: VectorBot, brain: Brain) -> dict:
    """Collect a small environment snapshot for the autonomous tick."""
    on_charger = None
    battery = "?"
    try:
        vector.get_battery_details()
        battery = vector.battery_level
        on_charger = bool(vector.robot.status.is_on_charger)
    except Exception as exc:
        print(f"[agent] state read failed: {exc}")
    idle_min = int((time.time() - brain.last_interaction) / 60)
    return {
        "battery": battery,
        "on_charger": on_charger,
        "time": time.strftime("%H:%M"),
        "idle_min": idle_min,
    }


def read_sensors(vector: VectorBot) -> tuple:
    """Return (is_being_touched, is_being_held) from Vector's sensors."""
    touched = held = False
    try:
        reading = vector.robot.touch.last_sensor_reading
        if reading is not None:
            touched = bool(reading.is_being_touched)
        held = bool(vector.robot.status.is_being_held)
    except Exception:
        pass
    return touched, held


def sensor_loop(
        ui: UserInterface,
        gpt: CustomGPT,
        vector: VectorBot,
        robot_action: Action,
        brain: Brain,
        voice: Voice,
        handler: StreamHandler,
    ) -> None:
    """Watch the backpack touch sensor and 'being held' state; react in character."""
    prev_touched = prev_held = False
    last_event = 0.0
    while True:
        time.sleep(0.2)
        touched, held = read_sensors(vector)
        event = None
        if touched and not prev_touched:
            event = "Someone is petting your back (your backpack touch sensor is being touched)."
        elif held and not prev_held:
            event = "Someone just picked you up off the ground."
        elif prev_held and not held:
            event = "You were just put back down on a surface."
        prev_touched, prev_held = touched, held
        if not event:
            continue

        now = time.time()
        if now < brain.convo_active_until:      # don't interrupt a conversation
            continue
        if now - last_event < TOUCH_COOLDOWN:   # debounce
            continue
        if not brain.lock.acquire(blocking=False):
            continue
        try:
            last_event = time.time()
            reply = gpt.event_tick(event)
            text, commands = parse_commands(reply)
            silent = any(c == "SILENT" for c in commands)
            commands = [c for c in commands if c not in ("LOOK", "SILENT")]
            if text and not silent:
                speak(text, robot_action, voice, handler)
                ui.add_text("Vector (touch)", text)
            if commands:
                robot_action.manage_commands(commands)
        except Exception as exc:
            print(f"[sensor] reaction failed: {exc}")
        finally:
            brain.lock.release()


def agent_loop(
        ui: UserInterface,
        gpt: CustomGPT,
        vector: VectorBot,
        robot_data: Data,
        robot_action: Action,
        brain: Brain,
        voice: Voice,
        handler: StreamHandler,
    ) -> None:
    """Autonomous loop: on an interval, sense the environment and let the LLM act."""
    while True:
        time.sleep(AGENT_INTERVAL)
        if not AUTONOMOUS:
            continue
        # Never interrupt an active conversation.
        if time.time() < brain.convo_active_until:
            continue
        if not brain.lock.acquire(blocking=False):
            continue
        try:
            state = gather_state(vector, brain)
            reply = gpt.autonomous_tick(state)
            _, commands = parse_commands(reply)

            # LLM asked to look around this tick.
            if any(c == "LOOK" for c in commands):
                frame = robot_data.get_pil_frame()
                if frame is not None:
                    reply = gpt.get_answer(
                        "This is your autonomous glance through your camera. "
                        "In one short sentence, react only if something is worth "
                        "mentioning; otherwise reply with @SILENT@.",
                        image=frame,
                    )

            text, commands = parse_commands(reply)
            silent = any(c == "SILENT" for c in commands)
            commands = [c for c in commands if c not in ("LOOK", "SILENT")]

            if text and not silent:
                speak(text, robot_action, voice, handler)
                ui.add_text("Vector (auto)", text)
            if commands:
                robot_action.manage_commands(commands)
        except Exception as exc:
            print(f"[agent] tick failed: {exc}")
        finally:
            brain.lock.release()


def main():
    # Initialise VectorBot
    vector = VectorBot()
    robot_action = Action(vector.robot)
    robot_data = Data(vector.robot)

    # Initialise the GPT brain and the Vietnamese voice
    gpt = CustomGPT()
    voice = Voice()

    # Listening source, in order of preference:
    #   1) Vector's OWN microphone over the SDK audio feed (what we want)
    #   2) a local/USB mic on the Pi via sounddevice
    #   3) no mic -> type to Vector in the UI
    handler = None
    try:
        from vectormic import VectorMicStream
        handler = VectorMicStream(vector.robot)
        print("[mic] using Vector's onboard microphone — say 'Vector' to talk")
    except Exception as exc:
        print(f"[mic] Vector mic unavailable ({exc}); trying a local mic")
        try:
            handler = StreamHandler()
        except Exception as exc2:
            print(f"[mic] no microphone ({exc2}); type to Vector in the UI instead")

    # Initialise UI
    ui = UserInterface()

    brain = Brain()

    # Route typed text straight to the brain (works with or without a mic).
    def handle_text(user_input: str) -> None:
        def worker():
            with brain.lock:
                brain.convo_active_until = time.time() + CONVO_WINDOW
                ui.add_text("Me", user_input)
                run_turn(
                    user_input,
                    want_frame=wants_vision(user_input),
                    gpt=gpt,
                    robot_data=robot_data,
                    robot_action=robot_action,
                    ui=ui,
                    voice=voice,
                    handler=handler,
                )
                brain.last_interaction = time.time()
        Thread(target=worker, daemon=True).start()

    ui.on_submit = handle_text

    # Startup sequence
    robot_action.emote('MessagingMessageGetIn')
    robot_action.emote('GreetAfterLongTime')
    speak("Tôi đã thức dậy rồi. Gọi tên tôi khi bạn cần nhé.", robot_action, voice, handler)
    robot_action.eyecolor(0.0, 0.0)

    # Active (wake-word) interaction thread — only if a microphone is present.
    if handler is not None:
        conversation_thread = Thread(
            target=conversation,
            args=(ui, handler, gpt, robot_data, robot_action, brain, voice),
        )
        conversation_thread.daemon = True
        conversation_thread.start()
        print("[wake] microphone ON — say 'Vector' to talk")
    else:
        print("[wake] no microphone — type to Vector in the UI window")

    # Autonomous agent-loop thread
    if AUTONOMOUS:
        agent_thread = Thread(
            target=agent_loop,
            args=(ui, gpt, vector, robot_data, robot_action, brain, voice, handler),
        )
        agent_thread.daemon = True
        agent_thread.start()
        print(f"[agent] autonomous loop ON (every {AGENT_INTERVAL:.0f}s)")
    else:
        print("[agent] autonomous loop OFF")

    # Touch / backpack sensor reaction thread
    if SENSORS:
        sensor_thread = Thread(
            target=sensor_loop,
            args=(ui, gpt, vector, robot_action, brain, voice, handler),
        )
        sensor_thread.daemon = True
        sensor_thread.start()
        print("[sensor] backpack touch reactions ON")

    print(f"[wake] listening for wake word; active window {CONVO_WINDOW:.0f}s")
    ui.start_ui()


if __name__ == "__main__":
    main()
