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

    def fake_extract(path, dst, max_minutes=None):
        t = np.arange(int(14 * SR)) / SR
        y = 0.1 * np.sin(2 * np.pi * 220 * t) * (1 + 0.3 * np.sin(2 * np.pi * 0.8 * t))
        import soundfile as sf

        sf.write(dst, y.astype(np.float32), SR)

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

