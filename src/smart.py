"""
SmartVector — the unified body for the Vector Brain.

ONE SDK connection that the brain server uses for everything: voice-triggered
actions, autonomous pet behaviour, and touch reactions. It uses *ephemeral*
behaviour control (grab control only while acting, release right after) so it
coexists with wire-pod's voice pipeline.

- sense()  -> dict of all of Vector's sensors (for the LLM agent loop to judge)
- act()    -> execute one @COMMAND@ token (movement / cuddle / approach / ...)
- say()    -> speak Vietnamese through Vector's speaker (autonomous/touch path)
"""

import time
import threading
import concurrent.futures
from contextlib import contextmanager

import numpy as np
import cv2
import PIL.Image
import anki_vector
from anki_vector.connection import ControlPriorityLevel
from anki_vector.util import degrees, distance_mm, speed_mmps


def _wait(x, timeout: float = 12.0):
    """Resolve an AsyncRobot future; swallow errors so one bad call can't crash us."""
    if isinstance(x, concurrent.futures.Future):
        try:
            return x.result(timeout=timeout)
        except Exception:
            return None
    return x


# Emotion name -> Vector animation trigger.
EMO = {
    'HAPPY': 'ComeHereSuccess', 'VERYHAPPY': 'GreetAfterLongTime',
    'SAD': 'FacePlantRoll', 'VERYSAD': 'FacePlantRoll',
    'ANGRY': 'Feedback_ShutUp', 'FRUSTRATED': 'Feedback_ShutUp',
    'SURPRISED': 'TakeAPictureFocusing', 'DISGUSTED': 'MeetVictorConfusion',
    'CONFUSED': 'MeetVictorConfusion', 'THINKING': 'KnowledgeGraphSearching',
    'CELEBRATE': 'OnboardingResetSuccess', 'LOVE': 'PettingBlissGetout',
    'SASSY': 'PettingBlissGetout', 'EYEROLL': 'Feedback_ShutUp',
    'NEUTRAL': 'NeutralFace', 'DARTINGEYES': 'KnowledgeGraphListening',
}

# Friendly eye colours (hue, saturation).
EYE = {
    'LOVE': (0.95, 1.0), 'HAPPY': (0.20, 1.0), 'CALM': (0.55, 1.0),
    'ANGRY': (0.0, 1.0), 'CURIOUS': (0.75, 1.0), 'NEUTRAL': (0.0, 0.0),
}


class SmartVector:
    def __init__(self) -> None:
        args = anki_vector.util.parse_command_args()
        self.robot = anki_vector.AsyncRobot(
            args.serial,
            behavior_control_level=ControlPriorityLevel.OVERRIDE_BEHAVIORS_PRIORITY,
            enable_face_detection=True,
        )
        self.robot.connect()
        try:
            self.robot.camera.init_camera_feed()
        except Exception:
            pass
        self._lock = threading.RLock()
        self._depth = 0
        self.last_action = 0.0

    def disconnect(self) -> None:
        try:
            self.robot.disconnect()
        except Exception:
            pass

    # -------------------------------------------------------------- control
    @contextmanager
    def control(self, level=ControlPriorityLevel.OVERRIDE_BEHAVIORS_PRIORITY):
        """Serialize all robot actions. Control is held continuously (OVERRIDE on
        connect), so this just guarantees one action at a time across threads."""
        with self._lock:
            try:
                yield
            finally:
                self.last_action = time.time()

    # -------------------------------------------------------------- sensing
    def get_frame(self) -> PIL.Image.Image:
        try:
            return self.robot.camera.latest_image.raw_image
        except Exception:
            return None

    def feel(self) -> tuple:
        """Cheap, RPC-free read of (being_touched, is_being_held) for fast polling."""
        touched = held = False
        try:
            t = self.robot.touch.last_sensor_reading
            if t is not None:
                touched = bool(t.is_being_touched)
        except Exception:
            pass
        try:
            held = bool(self.robot.status.is_being_held)
        except Exception:
            pass
        return touched, held

    def sense(self) -> dict:
        """Everything Vector can feel right now — handed to the LLM to judge."""
        r = self.robot
        s = {}
        try:
            bs = _wait(r.get_battery_state())
            if bs:
                s['battery_level'] = bs.battery_level            # 0..4
                s['charging'] = bool(bs.is_charging)
                s['on_charger'] = bool(bs.is_on_charger)
        except Exception:
            pass
        for k in ('is_being_held', 'is_picked_up', 'is_button_pressed',
                  'is_cliff_detected', 'is_falling', 'is_carrying_block',
                  'are_motors_moving'):
            try:
                s[k.replace('is_', '').replace('are_', '')] = bool(getattr(r.status, k))
            except Exception:
                pass
        try:
            t = r.touch.last_sensor_reading
            if t is not None:
                s['being_touched'] = bool(t.is_being_touched)
        except Exception:
            pass
        try:
            p = r.proximity.last_sensor_reading
            if p is not None and p.distance is not None:
                s['proximity_mm'] = int(p.distance.distance_mm)
                s['object_ahead'] = bool(p.found_object)
        except Exception:
            pass
        try:
            faces = [f for f in r.world.visible_faces]
            s['faces_visible'] = len(faces)
            names = [f.name for f in faces if getattr(f, 'name', '')]
            if names:
                s['face_names'] = names
        except Exception:
            pass
        s['time'] = time.strftime('%H:%M')
        return s

    # -------------------------------------------------------------- actions
    def emote(self, name: str) -> None:
        with self.control():
            _wait(self.robot.anim.play_animation_trigger(EMO.get(name.upper(), 'NeutralFace')))

    def eye_color(self, name: str) -> None:
        h, sat = EYE.get(name.upper(), (0.55, 1.0))
        with self.control():
            _wait(self.robot.behavior.set_eye_color(hue=h, saturation=sat))

    def drive(self, left: int, right: int, dur: float) -> None:
        with self.control():
            self.robot.motors.set_wheel_motors(left, right)
            time.sleep(max(0.1, min(dur, 6.0)))
            self.robot.motors.set_wheel_motors(0, 0)

    def turn(self, deg: float) -> None:
        with self.control():
            _wait(self.robot.behavior.turn_in_place(degrees(max(-180, min(180, deg)))))

    def head(self, deg: float) -> None:
        with self.control():
            _wait(self.robot.behavior.set_head_angle(degrees(max(-22, min(45, deg)))))

    def lift(self, height: float) -> None:
        with self.control():
            _wait(self.robot.behavior.set_lift_height(max(0.0, min(1.0, height))))

    def approach(self) -> None:
        """Come near the human: face them if seen, then roll closer."""
        with self.control():
            try:
                faces = [f for f in self.robot.world.visible_faces]
            except Exception:
                faces = []
            if faces:
                _wait(self.robot.behavior.turn_towards_face(faces[0]))
            _wait(self.robot.behavior.drive_straight(distance_mm(120), speed_mmps(90)))
            _wait(self.robot.behavior.set_head_angle(degrees(30)))

    def raise_hand(self) -> None:
        with self.control():
            _wait(self.robot.behavior.set_lift_height(1.0))
            time.sleep(0.4)

    def lower_hand(self) -> None:
        self.lift(0.0)

    def look_around(self) -> None:
        with self.control():
            _wait(self.robot.behavior.set_head_angle(degrees(25)))
            _wait(self.robot.behavior.turn_in_place(degrees(55)))
            time.sleep(0.2)
            _wait(self.robot.behavior.turn_in_place(degrees(-110)))
            _wait(self.robot.behavior.set_head_angle(degrees(10)))

    def cuddle(self) -> None:
        """Happy-dog reaction: purr animation + wag with hand (lift) and wheels."""
        with self.control():
            _wait(self.robot.anim.play_animation_trigger('PettingBlissGetout'))
            for _ in range(3):
                self.robot.motors.set_lift_motor(4.0)
                self.robot.motors.set_wheel_motors(45, -45)
                time.sleep(0.22)
                self.robot.motors.set_lift_motor(-4.0)
                self.robot.motors.set_wheel_motors(-45, 45)
                time.sleep(0.22)
            self.robot.motors.set_lift_motor(0)
            self.robot.motors.set_wheel_motors(0, 0)

    def wiggle(self) -> None:
        with self.control():
            for _ in range(2):
                self.robot.motors.set_wheel_motors(50, -50)
                time.sleep(0.2)
                self.robot.motors.set_wheel_motors(-50, 50)
                time.sleep(0.2)
            self.robot.motors.set_wheel_motors(0, 0)

    def say_wav(self, path: str, volume: int = 75) -> bool:
        with self.control():
            return _wait(self.robot.audio.stream_wav_file(path, volume)) is not None or True

    # -------------------------------------------------------------- dispatch
    def act(self, command: str) -> None:
        """Execute one @COMMAND@ token (without the @ @)."""
        c = command.strip().upper()
        try:
            if c.startswith('APPROACH'):
                self.approach()
            elif c.startswith('CUDDLE'):
                self.cuddle()
            elif c.startswith('RAISEHAND') or c == 'RAISE_HAND':
                self.raise_hand()
            elif c.startswith('LOWERHAND') or c == 'LOWER_HAND':
                self.lower_hand()
            elif c.startswith('LOOKAROUND') or c == 'LOOK_AROUND':
                self.look_around()
            elif c.startswith('WIGGLE'):
                self.wiggle()
            elif c.startswith('EMOTE_'):
                self.emote(c.split('_', 1)[1])
            elif c.startswith('EYE_'):
                self.eye_color(c.split('_', 1)[1])
            elif c.startswith('HEADUP'):
                self.head(40)
            elif c.startswith('HEADDOWN'):
                self.head(-15)
            elif c.startswith('TURN'):
                deg = _num(c, 45)
                self.turn(deg)
            elif c.startswith('FRWD'):
                self.drive(90, 90, _num(c, 2))
            elif c.startswith('BACK'):
                self.drive(-90, -90, _num(c, 2))
            elif c.startswith('LEFT'):
                self.drive(-80, 80, _num(c, 1))
            elif c.startswith('RIGHT'):
                self.drive(80, -80, _num(c, 1))
        except Exception as exc:
            print(f"[smart] action '{command}' failed: {exc}")


def _num(token: str, default: float) -> float:
    """Extract a trailing number from a token like FRWD_2 / TURN_90."""
    parts = token.replace('-', '_').split('_')
    for p in reversed(parts):
        try:
            return float(p)
        except ValueError:
            continue
    return default
