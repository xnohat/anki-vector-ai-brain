"""Minimal hardware smoke test: connect, see, speak Vietnamese. No driving."""
import sys, time
sys.path.insert(1, 'src')
from customgpt import _load_dotenv
_load_dotenv()

from vectorbot import VectorBot, Data, Action
from customgpt import CustomGPT
from voice import Voice

print("[1] Connecting to Vector ...")
vector = VectorBot()
action = Action(vector.robot)
data = Data(vector.robot)
print(f"    connected. battery_level={vector.battery_level}")

print("[2] Grabbing a camera frame (vision input) ...")
frame = None
for _ in range(8):
    frame = data.get_pil_frame()
    if frame is not None:
        break
    time.sleep(0.3)
print(f"    frame: {None if frame is None else frame.size}")

print("[3] Brain turn in Vietnamese (with vision) ...")
gpt = CustomGPT()
prompt = "Bạn nhìn thấy gì trước mặt? Trả lời thật ngắn."
reply = gpt.get_answer(prompt, image=frame)
print(f"    reply: {reply}")

print("[4] Speaking Vietnamese through Vector's speaker ...")
voice = Voice()
# strip @TOKENS@ for the spoken part
import re
spoken = re.sub(r'@.*?@', '', reply).strip() or "Xin chào, tôi là Vector."
wav = voice.synthesize(spoken)
print(f"    wav: {wav}")
ok = action.say_wav(wav) if wav else False
print(f"    playback ok: {ok}")

print("[5] Eye color blip + done.")
action.eyecolor(0.5, 1.0)
time.sleep(1)
vector.robot.disconnect()
print("DONE.")
