"""Feasibility test: can we stream Vector's OWN microphone over the SDK?

Calls the AudioFeed gRPC directly (the high-level enable_audio_feed is stubbed)
and reports whether real audio frames arrive. Speak near Vector while it runs.
"""
import sys, time, asyncio
sys.path.insert(1, 'src')
from customgpt import _load_dotenv
_load_dotenv()

import anki_vector
from anki_vector.connection import ControlPriorityLevel
from anki_vector.messaging import messages_pb2 as protocol

DURATION = 6.0

args = anki_vector.util.parse_command_args()
robot = anki_vector.AsyncRobot(
    args.serial,
    behavior_control_level=ControlPriorityLevel.OVERRIDE_BEHAVIORS_PRIORITY,
)
robot.connect()
print("connected. Speak near Vector now...")

frames = []

async def pull_audio():
    req = protocol.AudioFeedRequest()
    start = time.time()
    async for resp in robot.conn.grpc_interface.AudioFeed(req):
        frames.append(resp)
        if time.time() - start > DURATION:
            break

try:
    fut = robot.conn.run_coroutine(pull_audio())
    fut.result(timeout=DURATION + 5)
except Exception as exc:
    print(f"AudioFeed error: {type(exc).__name__}: {exc}")

print(f"\nframes received: {len(frames)}")
if frames:
    f = frames[len(frames)//2]
    sp = list(f.signal_power)
    print(f"per-frame signal_power samples: {len(sp)}")
    print(f"sample range: min={min(sp) if sp else '-'} max={max(sp) if sp else '-'}")
    print(f"source_direction={f.source_direction} confidence={f.source_confidence} "
          f"noise_floor={f.noise_floor_power}")
    total_samples = sum(len(list(fr.signal_power)) for fr in frames)
    print(f"total samples in {DURATION}s ~= {total_samples}  (=> ~{int(total_samples/DURATION)} Hz)")
else:
    print("No audio frames — this firmware/wire-pod does not expose the mic feed.")

robot.disconnect()
print("done.")
