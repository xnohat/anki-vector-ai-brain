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

import os
import ssl
import time
import socket
import threading
import configparser
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


_SDK_CONFIG = os.path.expanduser("~/.anki_vector/sdk_config.ini")


def _is_vector(ip: str, certfile: str, name: str, timeout: float = 2.0) -> bool:
    """True iff `ip:443` presents Vector's pinned self-signed cert (definitely him)."""
    try:
        ctx = ssl.create_default_context(cafile=certfile)
        ctx.check_hostname = False                # cert CN is the robot name, not the IP
        with socket.create_connection((ip, 443), timeout=timeout) as raw:
            with ctx.wrap_socket(raw, server_hostname=name):
                return True
    except Exception:
        return False


def _port_open(ip: str, timeout: float = 0.4) -> bool:
    try:
        with socket.create_connection((ip, 443), timeout=timeout):
            return True
    except Exception:
        return False


def discover_vector_ip() -> str:
    """Self-heal a DHCP IP change: if the IP in sdk_config.ini no longer answers as
    Vector, scan the local /24 for the host that presents his pinned cert, rewrite
    the config, and return the new IP. Returns None if discovery isn't possible.

    Vector connects OUT to wire-pod for voice, so voice keeps working even when his
    IP moves; but the SDK body connects IN to his IP, so a stale IP silently drops
    body control back to the firmware. This keeps the body following him."""
    try:
        cfg = configparser.ConfigParser()
        cfg.read(_SDK_CONFIG)
        if not cfg.sections():
            return None
        serial = cfg.sections()[0]
        cur_ip = cfg.get(serial, "ip", fallback=None)
        cert = cfg.get(serial, "cert", fallback=None)
        name = cfg.get(serial, "name", fallback="Vector")
        if not cert or not os.path.exists(cert):
            return cur_ip
        # Still where we think he is? Nothing to do.
        if cur_ip and _is_vector(cur_ip, cert, name):
            return cur_ip
        # Find our /24 and scan it for Vector's cert.
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("192.168.1.1", 9)); base = s.getsockname()[0]; s.close()
        except Exception:
            return cur_ip
        prefix = base.rsplit(".", 1)[0] + "."
        hosts = [prefix + str(i) for i in range(1, 255)]
        print(f"[smart] Vector not at {cur_ip}; scanning {prefix}0/24 for him...")
        open_hosts = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=64) as ex:
            for ip, ok in zip(hosts, ex.map(_port_open, hosts)):
                if ok:
                    open_hosts.append(ip)
        for ip in open_hosts:
            if _is_vector(ip, cert, name):
                cfg.set(serial, "ip", ip)
                with open(_SDK_CONFIG, "w") as fh:
                    cfg.write(fh)
                print(f"[smart] found Vector at {ip} (was {cur_ip}); config updated")
                return ip
        print("[smart] could not find Vector on the network (asleep/off?)")
        return cur_ip
    except Exception as exc:
        print(f"[smart] ip discovery error: {exc}")
        return None


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
    'CURIOUS': 'KnowledgeGraphSearching', 'INTERESTED': 'KnowledgeGraphSearching',
    # synonyms the LLM tends to use
    'EXCITED': 'OnboardingResetSuccess', 'PLAYFUL': 'ComeHereSuccess',
    'DIZZY': 'MeetVictorConfusion', 'SCARED': 'TakeAPictureFocusing',
    'WIDE': 'TakeAPictureFocusing', 'SHY': 'PettingBlissGetout',
}

# Friendly eye colours (hue, saturation).
EYE = {
    'LOVE': (0.95, 1.0), 'HAPPY': (0.20, 1.0), 'CALM': (0.55, 1.0),
    'ANGRY': (0.0, 1.0), 'CURIOUS': (0.75, 1.0), 'NEUTRAL': (0.0, 0.0),
}


class SmartVector:
    def __init__(self) -> None:
        # Self-heal a DHCP IP change before connecting: if his configured IP went
        # stale, find him on the LAN by his pinned cert and rewrite the config.
        try:
            discover_vector_ip()
        except Exception:
            pass
        args = anki_vector.util.parse_command_args()
        # Connect with retries: a previous (killed) process can leave stale
        # behaviour control on the robot for up to ~60s, which makes connect time
        # out. Retry until the robot frees control.
        last_exc = None
        for attempt in range(3):
            self.robot = anki_vector.AsyncRobot(
                args.serial,
                behavior_control_level=ControlPriorityLevel.OVERRIDE_BEHAVIORS_PRIORITY,
                enable_face_detection=True,
            )
            try:
                self.robot.connect()
                last_exc = None
                break
            except Exception as exc:
                last_exc = exc
                print(f"[smart] connect attempt {attempt + 1}/3 failed: {exc}")
                try:
                    self.robot.disconnect()
                except Exception:
                    pass
                time.sleep(6)
        if last_exc is not None:
            raise last_exc

        # CRITICAL: do NOT hold behaviour control while idle, or we fight wire-pod
        # (causes the robot to get stuck in the listening/"thinking" state with a
        # looping noise, and blocks voice/button). Release now; grab only to act.
        try:
            _wait(self.robot.conn.release_control(), timeout=8)
        except Exception:
            pass
        try:
            self.robot.camera.init_camera_feed()
        except Exception:
            pass
        self._lock = threading.RLock()
        self._depth = 0
        self.last_action = 0.0
        # Load Vector's built-in animation triggers (fist bump, wheelie, etc.) so
        # the agent can call them as built-in "intents".
        self.triggers = []
        try:
            r = self.robot.anim.load_animation_trigger_list()
            if isinstance(r, concurrent.futures.Future):
                r.result(timeout=8)
            self.triggers = list(self.robot.anim.anim_trigger_list)
        except Exception:
            pass

    def healthy(self) -> bool:
        """True if the SDK connection is live (a battery RPC returns)."""
        return self.battery()[0] is not None

    def disconnect(self) -> None:
        try:
            self.robot.disconnect()
        except Exception:
            pass

    # -------------------------------------------------------------- control
    @contextmanager
    def control(self, level=ControlPriorityLevel.OVERRIDE_BEHAVIORS_PRIORITY):
        """Ephemeral, reentrant behaviour control: request only while acting, then
        release so the robot is free for wire-pod voice/button. Serializes actions."""
        with self._lock:
            if self._depth == 0:
                _wait(self.robot.conn.request_control(level), timeout=8)
            self._depth += 1
            try:
                yield
            finally:
                self._depth -= 1
                if self._depth == 0:
                    _wait(self.robot.conn.release_control(), timeout=8)
                    self.last_action = time.time()

    # -------------------------------------------------------------- sensing
    def get_frame(self) -> PIL.Image.Image:
        try:
            return self.robot.camera.latest_image.raw_image
        except Exception:
            return None

    def feel(self) -> tuple:
        """Cheap, RPC-free read of (touched, held, button) for fast polling.

        `button` (backpack button pressed) means 'listen to me' — the caller must
        NOT react to it, so the robot's own voice/listening can run undisturbed.
        """
        touched = held = button = False
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
        try:
            button = bool(self.robot.status.is_button_pressed)
        except Exception:
            pass
        return touched, held, button

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
        try:
            a = r.accel
            s['accel'] = [round(a.x), round(a.y), round(a.z)]
            s['pitch_deg'] = round(r.pose_pitch_rad * 57.3)
        except Exception:
            pass
        s['time'] = time.strftime('%H:%M')
        return s

    def motion(self) -> tuple:
        """Cheap RPC-free (accel_z, gyro_magnitude, pitch_deg, tilt) for shake/flip.
        `tilt` is cos(angle from upright): 1.0 = level, 0.0 = on its side,
        -1.0 = upside down (catches a flip in ANY direction, not just roll)."""
        ax = ay = az = 0.0
        gmag = 0.0
        pitch = 0.0
        try:
            a = self.robot.accel
            ax, ay, az = float(a.x), float(a.y), float(a.z)
        except Exception:
            pass
        try:
            g = self.robot.gyro
            gmag = (g.x ** 2 + g.y ** 2 + g.z ** 2) ** 0.5
        except Exception:
            pass
        try:
            pitch = self.robot.pose_pitch_rad * 57.3
        except Exception:
            pass
        mag = (ax * ax + ay * ay + az * az) ** 0.5 or 1.0
        tilt = az / mag
        return az, gmag, pitch, tilt

    def cliff(self) -> bool:
        try:
            return bool(self.robot.status.is_cliff_detected)
        except Exception:
            return False

    def picked_up(self) -> bool:
        try:
            return bool(self.robot.status.is_picked_up)
        except Exception:
            return False

    def proximity(self) -> float:
        """Distance (mm) to whatever is in front of his face, or None. RPC-free."""
        try:
            p = self.robot.proximity.last_sensor_reading
            if p is not None and p.distance is not None:
                return float(p.distance.distance_mm)
        except Exception:
            pass
        return None

    def moving(self) -> bool:
        try:
            return bool(self.robot.status.are_wheels_moving)
        except Exception:
            return False

    def battery(self) -> tuple:
        """(level 0-3, is_charging, is_on_charger). RPC — call sparingly."""
        try:
            bs = _wait(self.robot.get_battery_state())
            if bs:
                return bs.battery_level, bool(bs.is_charging), bool(bs.is_on_charger)
        except Exception:
            pass
        return None, None, None

    def stop(self) -> None:
        try:
            self.robot.motors.stop_all_motors()
        except Exception:
            pass

    def return_to_charger(self) -> None:
        with self.control():
            try:
                _wait(self.robot.behavior.drive_on_charger())
            except Exception:
                pass

    # -------------------------------------------------------------- actions
    def emote(self, name: str) -> None:
        with self.control():
            _wait(self.robot.anim.play_animation_trigger(EMO.get(name.upper(), 'NeutralFace')))

    def play_trigger(self, name: str) -> bool:
        """Play a built-in Vector animation trigger (intent) by its real name."""
        if name not in self.triggers:
            return False
        with self.control():
            self._off_charger()
            _wait(self.robot.anim.play_animation_trigger(name))
        return True

    def eye_color(self, name: str) -> None:
        h, sat = EYE.get(name.upper(), (0.55, 1.0))
        with self.control():
            _wait(self.robot.behavior.set_eye_color(hue=h, saturation=sat))

    def _off_charger(self) -> None:
        """If docked, roll off the charger so the wheels can actually move."""
        try:
            if self.robot.status.is_on_charger:
                _wait(self.robot.behavior.drive_off_charger())
                time.sleep(0.3)
        except Exception:
            pass

    def _drive_safe(self, left: int, right: int, dur: float) -> bool:
        """Drive with raw motors but STOP + back off if a table edge (cliff) is
        seen. Returns True if an edge was hit. Raw motors bypass firmware cliff
        protection, so we must guard it ourselves."""
        self.robot.motors.set_wheel_motors(left, right)
        steps = max(1, int(min(dur, 6.0) / 0.08))
        for _ in range(steps):
            time.sleep(0.08)
            if self.cliff():
                self.robot.motors.set_wheel_motors(0, 0)
                self.robot.motors.set_wheel_motors(-80, -80)  # back away from edge
                time.sleep(0.5)
                self.robot.motors.set_wheel_motors(0, 0)
                return True
        self.robot.motors.set_wheel_motors(0, 0)
        return False

    def drive(self, left: int, right: int, dur: float) -> None:
        with self.control():
            self._off_charger()
            self._drive_safe(left, right, max(0.1, min(dur, 6.0)))

    def turn(self, deg: float) -> None:
        with self.control():
            _wait(self.robot.behavior.turn_in_place(degrees(max(-180, min(180, deg)))))

    def head(self, deg: float) -> None:
        with self.control():
            _wait(self.robot.behavior.set_head_angle(degrees(max(-22, min(45, deg)))))

    def lift(self, height: float) -> None:
        with self.control():
            _wait(self.robot.behavior.set_lift_height(max(0.0, min(1.0, height))))

    def _visible_faces(self) -> list:
        try:
            return [f for f in self.robot.world.visible_faces]
        except Exception:
            return []

    def find_caller(self) -> bool:
        """Turn to find and face the human who called (we have no sound direction,
        so rotate and look for a face — e.g. person behind -> Vector turns around)."""
        with self.control():
            self._off_charger()
            _wait(self.robot.behavior.set_head_angle(degrees(30)))
            for _ in range(9):                       # up to ~360 degrees
                faces = self._visible_faces()
                if faces:
                    _wait(self.robot.behavior.turn_towards_face(faces[0]))
                    return True
                _wait(self.robot.behavior.turn_in_place(degrees(45)))
                time.sleep(0.45)                     # let the camera/face detector update
            return False

    def approach(self) -> None:
        """Come near the human: find/face them, then roll closer."""
        with self.control():
            self._off_charger()
            faces = self._visible_faces()
            if faces:
                _wait(self.robot.behavior.turn_towards_face(faces[0]))
            else:
                self.find_caller()                   # person may be behind -> turn to find
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
            self._off_charger()
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

    def dance(self) -> None:
        """Dance: spin, bob the hand, wag — e.g. when music is heard."""
        with self.control():
            self._off_charger()
            _wait(self.robot.behavior.set_eye_color(hue=0.6, saturation=1.0))
            for i in range(4):
                self.robot.motors.set_wheel_motors(80, -80)
                self.robot.motors.set_lift_motor(5.0)
                time.sleep(0.3)
                self.robot.motors.set_wheel_motors(-80, 80)
                self.robot.motors.set_lift_motor(-5.0)
                time.sleep(0.3)
            self.robot.motors.set_wheel_motors(0, 0)
            self.robot.motors.set_lift_motor(0)

    # ------------------------------------------------------------ dog things
    def head_tilt(self) -> None:
        """Curious puppy head-tilt + look."""
        with self.control():
            _wait(self.robot.behavior.set_head_angle(degrees(20)))
            _wait(self.robot.anim.play_animation_trigger('MeetVictorConfusion'))
            _wait(self.robot.behavior.set_head_angle(degrees(5)))

    def spin(self) -> None:
        """Excited zoomies — a happy spin in place."""
        with self.control():
            self._off_charger()
            _wait(self.robot.behavior.set_eye_color(hue=0.2, saturation=1.0))
            _wait(self.robot.behavior.turn_in_place(degrees(360)))

    def nuzzle(self) -> None:
        """Roll up to the human and nuzzle/bump affectionately."""
        self.approach()
        with self.control():
            self._drive_safe(55, 55, 0.4)
            _wait(self.robot.anim.play_animation_trigger('PettingBlissGetout'))
            _wait(self.robot.behavior.set_head_angle(degrees(30)))

    def beg(self) -> None:
        """Sit up and beg — lift up, head up, hold, then down."""
        with self.control():
            _wait(self.robot.behavior.set_head_angle(degrees(40)))
            _wait(self.robot.behavior.set_lift_height(1.0))
            time.sleep(0.7)
            _wait(self.robot.behavior.set_lift_height(0.3))

    def perk_up(self) -> None:
        """Ears-up alert: head up, bright curious eyes, a listening perk."""
        with self.control():
            _wait(self.robot.behavior.set_head_angle(degrees(40)))
            _wait(self.robot.behavior.set_eye_color(hue=0.75, saturation=1.0))
            _wait(self.robot.anim.play_animation_trigger('KnowledgeGraphListening'))

    def wiggle(self) -> None:
        with self.control():
            self._off_charger()
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
            elif c.startswith('FINDME') or c.startswith('TURNTOME') or c.startswith('LOOKATME'):
                self.find_caller()
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
            elif c.startswith('HEADTILT'):
                self.head_tilt()
            elif c.startswith('SPIN') or c.startswith('ZOOMIES'):
                self.spin()
            elif c.startswith('NUZZLE'):
                self.nuzzle()
            elif c.startswith('BEG'):
                self.beg()
            elif c.startswith('PERKUP'):
                self.perk_up()
            elif c.startswith('DANCE'):
                self.dance()
            elif c.startswith('CHARGE') or c.startswith('GOCHARGE'):
                self.return_to_charger()
            elif c.startswith('STOP'):
                self.stop()
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
