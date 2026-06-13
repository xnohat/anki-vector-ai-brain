"""Live test: talk to Vector (Vietnamese) and see what his mic + Whisper hear.

Run it, then SPEAK NEAR VECTOR. Each finished utterance is transcribed and
printed. Ctrl+C to stop. Use this to confirm Vietnamese STT and to tune
VECTOR_MIC_THRESHOLD if needed.
"""
import sys, time
sys.path.insert(1, 'src')
from customgpt import _load_dotenv
_load_dotenv()

import anki_vector
from anki_vector.connection import ControlPriorityLevel
from vectormic import VectorMicStream

robot = anki_vector.AsyncRobot(
    anki_vector.util.parse_command_args().serial,
    behavior_control_level=ControlPriorityLevel.OVERRIDE_BEHAVIORS_PRIORITY,
)
robot.connect()
print("connected. Loading Whisper + starting mic...")
mic = VectorMicStream(robot)
print("\n>>> SPEAK VIETNAMESE NEAR VECTOR NOW (Ctrl+C to stop) <<<\n")

try:
    while True:
        if mic.stt_result:
            print(f"  HEARD: {mic.stt_result!r}")
            mic.stt_result = None
        time.sleep(0.2)
except KeyboardInterrupt:
    pass
finally:
    mic.running = False
    robot.disconnect()
    print("done.")
