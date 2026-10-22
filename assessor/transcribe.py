from __future__ import annotations

import os
from functools import lru_cache
from typing import Callable

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

    threads = max(1, (os.cpu_count() or 4) - 1) if device == "cpu" else 4
    return WhisperModel(size, device=device, compute_type=compute, cpu_threads=threads)


def transcribe(audio: str | np.ndarray, model_size: str = "base", device: str | None = None,
               language: str | None = None, progress: Callable[[str], None] | None = None) -> tuple[list[Segment], str | None]:
    """Transcribe audio with GPU acceleration (falling back to CPU) and live progress updates."""
    import gc
    gc.collect()

    dev, compute = (device, "int8" if device == "cpu" else "float16") if device else get_optimal_device()

    try:
        model = _model(model_size, dev, compute)
    except Exception:
        # Fallback to CPU if CUDA fails (e.g. driver incompatibility)
        dev, compute = "cpu", "int8"
        model = _model(model_size, dev, compute)

    segs_gen, info = model.transcribe(
        audio,
        language=language,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 300},
        beam_size=1,
        best_of=1,
        condition_on_previous_text=False,
    )

    total_dur = getattr(info, "duration", 0.0) or (len(audio) / 16000 if hasattr(audio, "__len__") else 0.0)
    out: list[Segment] = []
    last_update = 0.0

    for s in segs_gen:
        txt = s.text.strip()
        if txt:
            out.append(Segment(s.start, s.end, txt))
        if progress and total_dur > 0 and (s.end - last_update >= 20.0 or s.end >= total_dur - 1):
            last_update = s.end
            pct = min(99, int((s.end / total_dur) * 100))
            progress(f"Transcribing: {fmt(s.end)} / {fmt(total_dur)} ({pct}%)")

    gc.collect()
    return out, getattr(info, "language", None)
