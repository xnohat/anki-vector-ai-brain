"""
Vietnamese (and any-language) voice for Vector.

Vector's onboard TTS (behavior.say_text) is an English phoneme engine, so
Vietnamese comes out garbled. Instead we synthesize speech with OpenAI's TTS,
resample it to the WAV format Vector's speaker accepts (8000-16025 Hz, 16-bit,
mono) and stream it through the robot with robot.audio.stream_wav_file().

Config:
    VECTOR_TTS        - "1" to enable OpenAI voice (default), "0" to use onboard.
    VECTOR_TTS_MODEL  - OpenAI TTS model (default gpt-4o-mini-tts).
    VECTOR_TTS_VOICE  - voice name (default "alloy").
"""

import os
import io
import wave
import audioop  # stdlib in py3.11 (resampling)

from openai import OpenAI

TTS_ENABLED = os.environ.get("VECTOR_TTS", "1") not in ("0", "false", "False", "")
TTS_MODEL = os.environ.get("VECTOR_TTS_MODEL", "gpt-4o-mini-tts")
TTS_VOICE = os.environ.get("VECTOR_TTS_VOICE", "alloy")

# Vector wants 8000-16025 Hz; 16 kHz is the sweet spot for clarity.
TARGET_RATE = 16000
# OpenAI PCM output is 24 kHz, 16-bit, mono.
SOURCE_RATE = 24000


class Voice:
    def __init__(self, out_path: str = "tts_out.wav") -> None:
        self.enabled = TTS_ENABLED
        self.out_path = out_path
        self.client = OpenAI() if self.enabled else None

    def synthesize(self, text: str) -> str:
        """Synthesize `text` to a Vector-ready WAV file. Returns path, or None on failure."""
        if not self.enabled or not text:
            return None
        try:
            # Ask for raw 24kHz/16-bit/mono PCM so we can resample deterministically.
            resp = self.client.audio.speech.create(
                model=TTS_MODEL,
                voice=TTS_VOICE,
                input=text,
                response_format="pcm",
            )
            pcm24 = resp.read()  # bytes of 16-bit mono PCM @ 24kHz

            # Resample 24000 -> 16000 Hz (16-bit mono).
            pcm16, _ = audioop.ratecv(pcm24, 2, 1, SOURCE_RATE, TARGET_RATE, None)

            with wave.open(self.out_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(TARGET_RATE)
                wf.writeframes(pcm16)
            return self.out_path
        except Exception as exc:
            print(f"[voice] OpenAI TTS failed, will fall back: {exc}")
            return None
