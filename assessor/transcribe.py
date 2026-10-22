from __future__ import annotations

import math
import os
from functools import lru_cache
from typing import Callable

import numpy as np

from .types import Segment, fmt


def get_optimal_device() -> tuple[str, str]:
    """Auto-detect CUDA device and optimal compute type, falling back to CPU."""
    try:
        import ctranslate2
        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda", "float16"
    except Exception:
        pass
    return "cpu", "int8"


@lru_cache(maxsize=3)
def _model(size: str, device: str, compute: str):
    from faster_whisper import WhisperModel

    threads = min(4, max(1, (os.cpu_count() or 2) - 1)) if device == "cpu" else 4
    return WhisperModel(size, device=device, compute_type=compute, cpu_threads=threads)


def transcribe(audio: str | np.ndarray, model_size: str = "base", device: str | None = None,
               language: str | None = None, progress: Callable[[str], None] | None = None) -> tuple[list[Segment], str | None]:
    """Transcribe audio with GPU acceleration (falling back to CPU) and live progress updates."""
    import gc
    gc.collect()

    dev, compute = (device, "int8" if device == "cpu" else "float16") if device else get_optimal_device()

    def run(model, active_device: str) -> tuple[list[Segment], str | None]:
        if progress:
            action = "Detecting language and transcribing" if language is None else "Transcribing"
            progress(f"{action} with Whisper {model_size} on {active_device}")
        segs_gen, info = model.transcribe(
            audio,
            language=language,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 300},
            beam_size=1,
            best_of=1,
            condition_on_previous_text=False,
            word_timestamps=True,
        )

        total_dur = getattr(info, "duration", 0.0) or (len(audio) / 16000 if hasattr(audio, "__len__") else 0.0)
        out: list[Segment] = []
        last_update = 0.0

        for s in segs_gen:
            txt = s.text.strip()
            words = getattr(s, "words", None) or []
            if txt and words:
                phrase: list = []
                phrase_start = phrase_end = 0.0
                phrase_confidence: list[float] = []

                def flush_phrase() -> None:
                    if not phrase:
                        return
                    phrase_text = "".join(getattr(w, "word", "") for w in phrase).strip()
                    confidence = sum(phrase_confidence) / len(phrase_confidence) if phrase_confidence else None
                    if phrase_text:
                        out.append(Segment(phrase_start, phrase_end, phrase_text, confidence=confidence))
                    phrase.clear()
                    phrase_confidence.clear()

                for word in words:
                    if phrase and word.start - phrase_end > 0.5:
                        flush_phrase()
                    if not phrase:
                        phrase_start = word.start
                    phrase.append(word)
                    phrase_end = word.end
                    probability = getattr(word, "probability", None)
                    if probability is not None:
                        phrase_confidence.append(float(probability))
                flush_phrase()
            elif txt:
                confidence = getattr(s, "avg_logprob", None)
                confidence = float(math.exp(min(0.0, confidence))) if confidence is not None else None
                out.append(Segment(s.start, s.end, txt, confidence=confidence))
            if progress and total_dur > 0 and (s.end - last_update >= 20.0 or s.end >= total_dur - 1):
                last_update = s.end
                pct = min(99, int((s.end / total_dur) * 100))
                progress(f"Transcribing: {fmt(s.end)} / {fmt(total_dur)} ({pct}%)")

        return out, getattr(info, "language", None)

    if progress:
        progress(f"Loading Whisper {model_size} model on {dev}; first use may download model weights")
    try:
        model = _model(model_size, dev, compute)
    except Exception as exc:
        if dev == "cpu":
            raise RuntimeError(f"Could not load the Whisper {model_size} model: {exc}") from exc
        if progress:
            progress(f"GPU model initialization failed; retrying Whisper {model_size} on CPU")
        dev, compute = "cpu", "int8"
        model = _model(model_size, dev, compute)

    try:
        result = run(model, dev)
    except Exception as exc:
        if dev != "cuda":
            raise
        if progress:
            progress(f"GPU transcription failed ({type(exc).__name__}); retrying the full transcription on CPU")
        _model.cache_clear()
        dev, compute = "cpu", "int8"
        model = _model(model_size, dev, compute)
        result = run(model, dev)

    gc.collect()
    return result
