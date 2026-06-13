"""
Listen through Vector's OWN microphone (not a Pi mic).

The SDK's high-level enable_audio_feed is stubbed in this fork, but the raw
AudioFeed gRPC stream works: each AudioFeedResponse.signal_power is a chunk of
16 kHz, 16-bit, mono PCM (confirmed against wire-pod, which encodes Vector audio
as wav 16000/16/1). We stream it, run adaptive voice-activity detection, and
hand finished utterances to Whisper.

This class is a drop-in replacement for StreamHandler: it exposes the same
`.speaking`, `.stt_result`, and `.asst.talking` attributes app.py relies on.

Tuning (env):
    VECTOR_MIC_THRESHOLD   - min RMS to count as speech (default 500)
    VECTOR_MIC_FACTOR      - speech must exceed noise_floor * factor (default 2.5)
    VECTOR_MIC_SILENCE     - seconds of silence that ends an utterance (default 0.7)
    VECTOR_MIC_MINSPEECH   - min seconds of speech to bother transcribing (default 0.4)
"""

import os
import wave
import queue
from threading import Thread

import numpy as np

from anki_vector.messaging import messages_pb2 as protocol
from whisperstt import WhisperSTT

SAMPLE_RATE = 16000          # confirmed from wire-pod (wav.NewEncoder(out,16000,16,1,1))
WAV_PATH = "dictate.wav"


class _Asst:
    running = True
    talking = False
    analyze = None


class VectorMicStream:
    def __init__(self, robot) -> None:
        self.robot = robot
        self.asst = _Asst()
        self.speaking = False
        self.stt_result = None
        self.running = True

        self.min_threshold = float(os.environ.get("VECTOR_MIC_THRESHOLD", "500"))
        self.factor = float(os.environ.get("VECTOR_MIC_FACTOR", "2.5"))
        self.silence_sec = float(os.environ.get("VECTOR_MIC_SILENCE", "0.7"))
        self.min_speech_sec = float(os.environ.get("VECTOR_MIC_MINSPEECH", "0.4"))

        self._q: queue.Queue = queue.Queue()
        self._noise = 700.0   # adaptive room-noise floor

        # IMPORTANT: on current wire-pod firmware the SDK AudioFeed is a stub that
        # emits a synthetic 1000 Hz test tone instead of the real microphone. Probe
        # it first; if it's the placeholder, refuse so app.py falls back to a USB
        # mic / text. If a future firmware enables the real mic this will pass.
        if self._is_stub_tone():
            raise RuntimeError(
                "Vector SDK audio feed is a firmware stub (1 kHz test tone), "
                "not the real microphone"
            )

        self.whisper = WhisperSTT()
        # Start the audio feed on the robot connection loop, and a worker thread
        # that does VAD + transcription (keeps the async loop drop-free).
        self._fut = self.robot.conn.run_coroutine(self._feed())
        Thread(target=self._process, daemon=True).start()
        print("\033[32mListening through Vector's microphone..\033[0m")

    def _is_stub_tone(self) -> bool:
        """Capture a short sample and detect the synthetic pure-tone placeholder."""
        frames = []

        async def grab():
            req = protocol.AudioFeedRequest()
            async for resp in self.robot.conn.grpc_interface.AudioFeed(req):
                frames.append(resp.signal_power)
                if len(frames) >= 6:
                    break

        try:
            self.robot.conn.run_coroutine(grab()).result(timeout=6)
        except Exception as exc:
            raise RuntimeError(f"audio feed unavailable: {exc}")

        x = np.frombuffer(b"".join(frames), dtype=np.int16).astype(np.float64)
        if x.size < 1024:
            return False
        x = x - x.mean()
        spec = np.abs(np.fft.rfft(x))
        spec[0] = 0.0
        total = float(np.sum(spec ** 2))
        if total <= 0:
            return True
        # A real room is broadband; the stub is ~all energy in one bin.
        purity = float(spec.max() ** 2 / total)
        return purity > 0.85

    async def _feed(self) -> None:
        """Pull raw mic chunks; do the absolute minimum here to avoid drops."""
        req = protocol.AudioFeedRequest()
        try:
            async for resp in self.robot.conn.grpc_interface.AudioFeed(req):
                if not self.running:
                    break
                if self.asst.talking:
                    continue  # ignore frames while Vector is speaking (no echo)
                self._q.put(resp.signal_power)
        except Exception as exc:
            print(f"[vectormic] audio feed ended: {exc}")

    def _process(self) -> None:
        voiced = bytearray()
        in_speech = False
        silence = 0.0
        speech = 0.0
        while self.running:
            try:
                chunk = self._q.get(timeout=0.5)
            except queue.Empty:
                self.speaking = False
                continue
            samples = np.frombuffer(chunk, dtype=np.int16)
            if samples.size == 0:
                continue
            dt = samples.size / SAMPLE_RATE
            rms = float(np.sqrt(np.mean(samples.astype(np.float32) ** 2)))
            trigger = max(self.min_threshold, self._noise * self.factor)

            if rms > trigger:
                self.speaking = True
                in_speech = True
                silence = 0.0
                speech += dt
                voiced += chunk
            else:
                # adapt the noise floor only when not actively in speech
                self._noise = 0.95 * self._noise + 0.05 * rms
                if in_speech:
                    voiced += chunk            # keep a little trailing audio
                    silence += dt
                    if silence >= self.silence_sec:
                        self.speaking = False
                        if speech >= self.min_speech_sec:
                            self._transcribe(bytes(voiced))
                        voiced, in_speech, silence, speech = bytearray(), False, 0.0, 0.0
                else:
                    self.speaking = False

    def _transcribe(self, pcm: bytes) -> None:
        try:
            with wave.open(WAV_PATH, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(pcm)
            text = (self.whisper.inference() or "").strip()
            if text:
                self.stt_result = text
        except Exception as exc:
            print(f"[vectormic] transcription failed: {exc}")
