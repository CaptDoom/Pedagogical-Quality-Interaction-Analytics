"""Coverage for media download (link path) and graceful LLM failure.

- download() must fetch a recording over plain HTTP (the fallback branch after gdown/yt-dlp).
- llm.run() must surface connection errors so the pipeline can catch them, and pipeline.analyze()
  must still return a full report (measured signals only, explanatory note) when the LLM backend is down.
"""
from __future__ import annotations

import http.server
import os
import threading

import numpy as np
import pytest

from assessor import pipeline as pl
from assessor.llm import run as llm_run
from assessor.media import download
from assessor.transcribe import transcribe
from assessor.types import Segment

SR = 16000


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):  # keep pytest output clean
        pass


def _serve(directory) -> http.server.ThreadingHTTPServer:
    handler = lambda *a, **k: _QuietHandler(*a, directory=str(directory), **k)  # noqa: E731
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def test_download_plain_http(tmp_path):
    payload = b"classlens-recording-bytes" * 512
    (tmp_path / "rec.mp4").write_bytes(payload)
    srv = _serve(tmp_path)
    try:
        url = f"http://127.0.0.1:{srv.server_address[1]}/rec.mp4"
        out = download(url, str(tmp_path / "dl"))
        assert os.path.exists(out), "download() returned a missing file"
        with open(out, "rb") as fh:
            assert fh.read() == payload, "downloaded bytes differ from source"
    finally:
        srv.shutdown()


def test_llm_run_surfaces_connection_error(monkeypatch):
    """llm.run must raise (not swallow) so the pipeline's catch can add its note."""
    monkeypatch.setenv("OLLAMA_HOST", "http://127.0.0.1:1")  # nothing listens here
    segs = [Segment(0.0, 4.0, "Great job! What do plants need to grow?", "teacher"),
            Segment(5.0, 8.0, "They need water and sunlight", "student")]
    feats = {"duration_s": 8.0, "student_share": 0.4, "student_turns": 1, "questions_per_min": 1.5,
             "open_question_share": 1.0, "median_wait_s": 1.0, "teacher_wpm": 120.0,
             "praise_per_min": 0.5, "dead_air_ratio": 0.0, "longest_teacher_monologue_s": 4.0}
    with pytest.raises(Exception):
        llm_run("ollama", segs, feats)


def test_llm_http_error_includes_body():
    """HTTP error bodies (e.g. insufficient_quota) must surface, not just the status code."""
    from assessor.llm import _post

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            cl = int(self.headers.get("Content-Length", 0))
            if cl > 0:
                self.rfile.read(cl)
            body = b'{"error":{"message":"You have no credits remaining","code":"credit_balance_exhausted"}}'
            self.send_response(429)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{srv.server_address[1]}/v1/chat"
        with pytest.raises(RuntimeError) as ei:
            _post(url, {"m": 1}, {})
        msg = str(ei.value)
        assert "429" in msg and "no credits" in msg
    finally:
        srv.shutdown()


def test_pipeline_degrades_when_llm_fails(tmp_path, monkeypatch):
    """Full analyze() run: transcription/audio are stubbed, LLM backend is down -> report + note, no crash."""
    monkeypatch.setenv("OLLAMA_HOST", "http://127.0.0.1:1")
    monkeypatch.setattr(pl, "has_ffmpeg", lambda: True)

    src = tmp_path / "input.mp4"
    src.write_bytes(b"not-really-a-video")

    def fake_extract(path, dst, max_minutes=None, denoised_dst=None, denoise=False):
        t = np.arange(int(14 * SR)) / SR
        y = 0.1 * np.sin(2 * np.pi * 220 * t) * (1 + 0.3 * np.sin(2 * np.pi * 0.8 * t))
        import soundfile as sf

        sf.write(dst, y.astype(np.float32), SR)
        if denoised_dst:
            sf.write(denoised_dst, y.astype(np.float32), SR)

    segs = [Segment(0.0, 4.0, "Great job class! What do plants need in order to grow?", "teacher"),
            Segment(5.0, 8.0, "Plants need water and sunlight to grow", "student"),
            Segment(9.0, 12.0, "Exactly! Well done.", "teacher")]
    monkeypatch.setattr(pl, "extract_audio", fake_extract)
    monkeypatch.setattr(pl, "ingest", lambda src, outdir: (
        src, {"duration": 14.0, "has_audio": True, "has_video": False, "format": "test"}))
    monkeypatch.setattr(pl, "transcribe", lambda *a, **k: (segs, "en"))

    report = pl.analyze(str(src), "tiny", False, "ollama", 60)

    assert report["label"] in ("Poor", "Average", "Good", "Excellent")
    assert any("LLM review unavailable" in n for n in report["notes"]), report["notes"]
    assert [d for d in report["dimensions"] if d["source"] == "LLM"] == [], "LLM dims must be absent on failure"
    assert report["observations"], "measured-signal observations must still be produced"


def test_gemini_backend_routing(monkeypatch):
    """Test gemini backend key validation and routing."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    segs = [Segment(0.0, 4.0, "What is photosynthesis?", "teacher")]
    feats = {"duration_s": 4.0, "student_share": 0.0, "student_turns": 0, "questions_per_min": 15.0,
             "open_question_share": 1.0, "median_wait_s": None, "teacher_wpm": 120.0,
             "praise_per_min": 0.0, "dead_air_ratio": 0.0, "longest_teacher_monologue_s": 4.0}

    # No key provided
    with pytest.raises(RuntimeError, match="API key is not set"):
        llm_run("gemini", segs, feats)

    # API key provided directly
    monkeypatch.setattr("assessor.llm._post", lambda *a, **k: {
        "candidates": [{"content": {"parts": [{"text": '{"clarity": 4, "rapport": 4, "content_accuracy": 5, "summary": "Clear question asked.", "strengths": [], "weaknesses": []}'}]}}]
    })
    res = llm_run("gemini", segs, feats, api_key="AIzaSyDummyTestKey123")
    assert res is not None
    assert res["clarity"] == 4
    assert res["content_accuracy"] == 5


def test_transcribe_detects_language_in_one_pass_and_splits_word_pauses(monkeypatch):
    from types import SimpleNamespace

    calls = []

    class FakeModel:
        def transcribe(self, audio, language=None, **kwargs):
            calls.append((language, kwargs))
            info = SimpleNamespace(language="ml", language_probability=0.95, duration=3.0)
            words = [
                SimpleNamespace(start=0.0, end=0.3, word=" Hello", probability=0.9),
                SimpleNamespace(start=0.35, end=0.7, word=" there", probability=0.8),
                SimpleNamespace(start=1.4, end=1.8, word=" friend", probability=0.7),
            ]
            segment = SimpleNamespace(start=0.0, end=1.8, text=" Hello there friend", words=words)
            return iter([segment]), info

    monkeypatch.setattr("assessor.transcribe._model", lambda *args: FakeModel())

    segments, language = transcribe(np.zeros(3 * SR, dtype=np.float32), "tiny")

    assert language == "ml"
    assert len(calls) == 1
    assert calls[0][0] is None
    assert calls[0][1]["word_timestamps"] is True
    assert [segment.text for segment in segments] == ["Hello there", "friend"]
    assert segments[0].confidence == pytest.approx(0.85)
    assert segments[1].confidence == pytest.approx(0.7)


def test_transcribe_honors_explicit_language(monkeypatch):
    from types import SimpleNamespace

    calls = []

    class FakeModel:
        def transcribe(self, audio, language=None, **kwargs):
            calls.append(language)
            info = SimpleNamespace(language=language, duration=3.0)
            segment = SimpleNamespace(start=0.0, end=0.5, text=" Hello", words=[])
            return iter([segment]), info

    monkeypatch.setattr("assessor.transcribe._model", lambda *args: FakeModel())

    segments, language = transcribe(np.zeros(3 * SR, dtype=np.float32), "tiny", language="ml")

    assert calls == ["ml"]
    assert language == "ml"
    assert [segment.text for segment in segments] == ["Hello"]


def test_transcribe_retries_cuda_failure_on_cpu(monkeypatch):
    from types import SimpleNamespace

    class FakeModel:
        def __init__(self, device):
            self.device = device

        def transcribe(self, audio, language=None, **kwargs):
            info = SimpleNamespace(language="en", language_probability=0.95, duration=3.0)

            def segments():
                if self.device == "cuda":
                    raise RuntimeError("CUDA driver unavailable")
                word = SimpleNamespace(start=0.0, end=0.5, word=" Hello", probability=0.9)
                yield SimpleNamespace(start=0.0, end=0.5, text=" Hello", words=[word])

            return segments(), info

    monkeypatch.setattr("assessor.transcribe.get_optimal_device", lambda: ("cuda", "float16"))

    def fake_model(size, device, compute):
        return FakeModel(device)

    fake_model.cache_clear = lambda: None
    monkeypatch.setattr("assessor.transcribe._model", fake_model)
    updates = []

    segments, language = transcribe(
        np.zeros(3 * SR, dtype=np.float32), "tiny", progress=updates.append
    )

    assert language == "en"
    assert [segment.text for segment in segments] == ["Hello"]
    assert any("retrying the full transcription on CPU" in message for message in updates)


def test_pipeline_rejects_recordings_without_detected_speech(tmp_path, monkeypatch):
    monkeypatch.setattr(pl, "has_ffmpeg", lambda: True)
    src = tmp_path / "silent.mp4"
    src.write_bytes(b"test media")

    def fake_extract(path, dst, max_minutes=None, denoised_dst=None, denoise=False):
        import soundfile as sf

        silence = np.zeros(3 * SR, dtype=np.float32)
        sf.write(dst, silence, SR)
        if denoised_dst:
            sf.write(denoised_dst, silence, SR)

    monkeypatch.setattr(pl, "extract_audio", fake_extract)
    monkeypatch.setattr(pl, "ingest", lambda path, outdir: (
        path, {"duration": 3.0, "has_audio": True, "has_video": False, "format": "test"}))
    monkeypatch.setattr(pl, "transcribe", lambda *args, **kwargs: ([], "en"))

    with pytest.raises(RuntimeError, match="No speech was detected"):
        pl.analyze(str(src), "tiny", False, "none")


def test_pipeline_skips_speaker_model_when_diarization_disabled(tmp_path, monkeypatch):
    monkeypatch.setattr(pl, "has_ffmpeg", lambda: True)
    src = tmp_path / "input.mp4"
    src.write_bytes(b"test media")
    denoise_options = []

    def fake_extract(path, dst, max_minutes=None, denoised_dst=None, denoise=False):
        import soundfile as sf

        denoise_options.append(denoise)
        audio = np.sin(2 * np.pi * 220 * np.arange(5 * SR) / SR).astype(np.float32) * 0.1
        sf.write(dst, audio, SR)
        if denoised_dst:
            sf.write(denoised_dst, audio, SR)

    segments = [Segment(0, 2, "What do you see?", "teacher"), Segment(2.5, 4, "I see a cat.", "teacher")]
    monkeypatch.setattr(pl, "extract_audio", fake_extract)
    monkeypatch.setattr(pl, "ingest", lambda path, outdir: (
        path, {"duration": 5.0, "has_audio": True, "has_video": False, "format": "test"}))
    monkeypatch.setattr(pl, "transcribe", lambda *args, **kwargs: (segments, "en"))
    monkeypatch.setattr(pl, "assign_speakers", lambda *args, **kwargs: pytest.fail("diarization should be skipped"))

    report = pl.analyze(str(src), "tiny", False, "none", 5, use_diarization=False)

    assert report["diarization"]["method"] == "disabled"
    assert report["diarization"]["confident"] is False
    assert any("Speaker separation was disabled" in note for note in report["notes"])
    assert denoise_options == [True]
