"""HTTP route and request validation coverage for the FastAPI service."""
import time

from fastapi.testclient import TestClient

import server


client = TestClient(server.app)


def test_read_routes_respond(monkeypatch):
    monkeypatch.delenv("APP_PASSWORD", raising=False)
    assert client.get("/").status_code == 200
    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/api/health").json()["app"] == "ClassLens"
    assert client.get("/api/config").json() == {
        "whisper_model": "base", "llm_backend": "none", "speaker_diarization": False
    }
    assert client.get("/static/styles.css").status_code == 200
    assert client.get("/favicon.ico").status_code == 204
    samples = client.get("/api/samples").json()
    assert isinstance(samples, list)
    assert all({"name", "rel_path", "size_mb", "ext"} <= sample.keys() for sample in samples)
    latest = client.get("/api/latest")
    assert latest.status_code == 200
    assert isinstance(latest.json(), dict)


def test_unknown_job_and_media_return_404():
    assert client.get("/jobs/does-not-exist").status_code == 404
    assert client.get("/api/media", params={"path": "../../../../Windows/win.ini"}).status_code == 404


def test_assess_requires_input_and_rejects_unknown_backend():
    missing = client.post("/assess", data={"llm_backend": "none"})
    assert missing.status_code == 400
    assert "Provide a file" in missing.json()["detail"]

    bad_backend = client.post("/assess", data={"url": "https://example.invalid/lesson.mp4", "llm_backend": "unknown"})
    assert bad_backend.status_code == 400
    assert "Unknown LLM backend" in bad_backend.json()["detail"]


def test_assess_validates_duration_and_sample_type():
    duration = client.post("/assess", data={"url": "https://example.invalid/lesson.mp4", "max_minutes": "0"})
    assert duration.status_code == 422

    unsupported_sample = client.post("/assess", data={"sample": "config.json", "llm_backend": "none"})
    assert unsupported_sample.status_code == 400
    assert "audio or video" in unsupported_sample.json()["detail"]


def test_assess_enforces_configured_password(monkeypatch):
    monkeypatch.setenv("APP_PASSWORD", "test-password")
    response = client.post("/assess", data={"url": "https://example.invalid/lesson.mp4", "token": "wrong"})
    assert response.status_code == 401
    assert response.json()["detail"] == "Wrong or missing password."


def test_report_routes_enforce_configured_password(monkeypatch):
    monkeypatch.setenv("APP_PASSWORD", "test-password")
    monkeypatch.setattr(server, "LATEST_REPORT", {"label": "Good"})
    monkeypatch.setitem(server.JOBS, "protected", {"status": "done"})

    assert client.get("/api/latest").status_code == 401
    assert client.get("/jobs/protected").status_code == 401
    headers = {"X-App-Password": "test-password"}
    assert client.get("/api/latest", headers=headers).json() == {"label": "Good"}
    assert client.get("/jobs/protected", headers=headers).json()["status"] == "done"


def test_assessment_job_lifecycle(monkeypatch):
    monkeypatch.delenv("APP_PASSWORD", raising=False)
    analysis_args = []

    def fake_analyze(*args, **kwargs):
        analysis_args.extend(args)
        return {"label": "Good", "score": 3.8, "features": {}, "dimensions": [], "observations": []}

    monkeypatch.setattr(server, "analyze", fake_analyze)
    response = client.post(
        "/assess", data={"llm_backend": "none"}, files={"file": ("lesson.wav", b"test-audio", "audio/wav")}
    )
    assert response.status_code == 200
    job_id = response.json()["job"]

    job = None
    for _ in range(100):
        job = client.get(f"/jobs/{job_id}").json()
        if job["status"] in {"done", "error"}:
            break
        time.sleep(0.01)

    assert job["status"] == "done", job
    assert analysis_args[1] == "base"
    assert analysis_args[2] is False
    assert analysis_args[4] == 15
    assert job["report"]["file_name"] == "lesson.wav"
    assert client.get("/api/latest").json()["label"] == "Good"
