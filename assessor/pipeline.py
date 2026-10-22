from __future__ import annotations

import os
import shutil
import tempfile
from typing import Callable

from . import llm as llm_mod
from .features import apply_interaction_tags, assign_speakers, compute_features
from .media import extract_audio, has_ffmpeg, ingest, sample_frames
from .scoring import assess
from .transcribe import transcribe
from .types import fmt


def analyze(path: str, whisper_model: str = "medium", use_frames: bool = True, llm_backend: str = "none",
            max_minutes: float = 60, progress: Callable[[str], None] | None = None, device: str | None = None,
            api_key: str | None = None) -> dict:
    """Full pipeline: audio -> transcript -> speakers -> features -> rubric (+ optional LLM) -> report.
    All intermediate files live in a temp dir that is deleted afterwards."""
    say = progress or (lambda m: None)
    if not has_ffmpeg():
        raise RuntimeError("ffmpeg is required but was not found on PATH.")
    tmp = tempfile.mkdtemp(prefix="classlens_")
    notes = []
    try:
        say("Downloading link" if not os.path.isfile(path) else "Checking recording")
        path, info = ingest(path, tmp)
        if info and info.get("duration", 0) > max_minutes * 60:
            notes.append(f"Recording is {info['duration']/60:.0f} min; only the first {max_minutes:.0f} minutes were analysed.")

        say("Extracting audio")
        wav = os.path.join(tmp, "audio.wav")
        denoised_wav = os.path.join(tmp, "audio_denoised.wav")
        extract_audio(path, wav, max_minutes, denoised_dst=denoised_wav)
        import gc
        import soundfile as sf
        y, sr = sf.read(wav, dtype="float32")
        clean_y, clean_sr = sf.read(denoised_wav, dtype="float32")
        if clean_sr != sr:
            raise RuntimeError(f"Audio sample-rate mismatch after noise reduction ({sr} Hz vs {clean_sr} Hz).")
        gc.collect()

        say("Transcribing locally")
        segs, lang = transcribe(clean_y, whisper_model, device, progress=say)
        gc.collect()
        if not segs:
            raise RuntimeError(
                "No speech was detected in the selected audio. Check that it contains audible speech, "
                "increase the analysis duration, or try another Whisper model."
            )

        say("Separating teacher and student voices")
        segs, dia = assign_speakers(segs, clean_y, sr, progress=say)
        if dia.get("error"):
            notes.append("Speaker separation was unavailable; participation estimates may be unreliable.")

        frames, step = ([], 10.0)
        if use_frames and info.get("has_video", False):
            say("Sampling screen frames")
            frames, step = sample_frames(path, 10.0, 150, max_minutes * 60)

        say("Measuring speech activity, interaction, delivery and screen signals")
        feats = compute_features(segs, y, sr, frames, step, dia["confident"], vad_audio=clean_y)

        llm_out = None
        if llm_backend not in (None, "", "none") and segs:
            say(f"Asking {llm_backend} for qualitative review")
            try:
                llm_out = llm_mod.run(llm_backend, segs, feats, api_key=api_key)
            except Exception as e:  # noqa: BLE001
                detail = str(e).replace("\n", " ")[:260]
                notes.append(f"LLM review unavailable ({type(e).__name__}: {detail}); "
                             "score uses measured signals only.")

        say("Scoring")
        if llm_out:
            apply_interaction_tags(feats, llm_out.get("interaction_tags"), segs)
        report = assess(feats, llm_out)
        report["evidence"] = feats.get("_evidence", {})
        report["notes"] = notes
        report["language"] = lang
        report["diarization"] = dia
        report["transcript"] = [
            {"t": fmt(s.start), "start": round(s.start, 2), "end": round(s.end, 2),
             "speaker": s.speaker, "confidence": round(s.confidence, 3) if s.confidence is not None else None,
             "text": s.text}
            for s in segs
        ]
        return report
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
