import os
import whisper

# Language for speech-to-text. Default Vietnamese (the user speaks Vietnamese).
# Override with VECTOR_STT_LANG (e.g. "en", "vi"). Empty => Whisper auto-detect.
STT_LANG = os.environ.get("VECTOR_STT_LANG", "vi").strip()
# 'small' is multilingual (supports Vietnamese). 'base' is faster but weaker.
STT_MODEL = os.environ.get("VECTOR_STT_MODEL", "small").strip()


class WhisperSTT:
    def __init__(self, model_size: str = None, language: str = None, translate: bool = False) -> None:
        self.language = (language if language is not None else STT_LANG) or None
        self.translate = translate
        model_size = model_size or STT_MODEL
        print(f"\033[96mLoading Whisper '{model_size}' (lang={self.language or 'auto'})..\033[0m",
              end='', flush=True)
        self.model = whisper.load_model(model_size)
        print("\033[90m Done.\033[0m")

    def inference(self, path: str = 'dictate.wav') -> str:
        audio = whisper.load_audio(path)
        audio = whisper.pad_or_trim(audio)

        mel = whisper.log_mel_spectrogram(audio).to(self.model.device)

        options = whisper.DecodingOptions(
            language=self.language,                       # None => auto-detect
            task="translate" if self.translate else "transcribe",
        )
        result = whisper.decode(self.model, mel, options)
        return result.text
