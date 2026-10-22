"""ClassLens HTTP service (FastAPI).

Run:  uvicorn server:app --port 8000      (or: python server.py)
UI:   http://localhost:8000/              high-fidelity pedagogical dashboard and ingestion modal.

POST /assess starts a job (file upload, link, or local sample; settings as form fields) and returns {job}.
GET  /jobs/{id} returns {status: queued|running|done|error, messages, report, error}.
GET  /api/samples returns available demo recordings on disk.
Set APP_PASSWORD to require it as the "token" field on every /assess call.
Only transcript text and metrics ever leave this process (LLM backends); audio stays local.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import threading
import uuid
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from assessor.env import load_env

load_env()

from assessor.pipeline import analyze  # noqa: E402  (env must be loaded first)

app = FastAPI(title="ClassLens", docs_url="/api")

ROOT_DIR = Path(__file__).resolve().parent
WEB_PAGE = ROOT_DIR / "web" / "index.html"
WEB_DIR = ROOT_DIR / "web"
WHISPER_MODELS = {"tiny", "base", "small", "medium"}
LLM_BACKENDS = {"none", "gemini", "google", "openai-compatible", "ollama", "anthropic"}
DEFAULT_WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "small")
DEFAULT_LLM_BACKEND = os.environ.get("DEFAULT_LLM_BACKEND", "gemini")
if DEFAULT_WHISPER_MODEL not in WHISPER_MODELS:
    raise RuntimeError(f"Unsupported WHISPER_MODEL {DEFAULT_WHISPER_MODEL!r}.")
if DEFAULT_LLM_BACKEND not in LLM_BACKENDS:
    raise RuntimeError(f"Unsupported DEFAULT_LLM_BACKEND {DEFAULT_LLM_BACKEND!r}.")
JOBS: dict[str, dict] = {}
LATEST_REPORT: dict | None = None
_LOCK = threading.Lock()  # one analysis at a time: whisper already saturates the CPU


def _is_within_root(path: Path) -> bool:
    """Return whether a resolved path is inside the application directory."""
    try:
        path.resolve().relative_to(ROOT_DIR)
        return True
    except ValueError:
        return False


def _serialize(report: dict) -> dict:
    return json.loads(json.dumps(report, default=float))


def _require_password(x_app_password: str = Header(default="")) -> None:
    required = os.environ.get("APP_PASSWORD")
    if required and x_app_password != required:
        raise HTTPException(status_code=401, detail="Wrong or missing password.")


def _run(job: dict, source: str, saved_path: str | None, model: str, use_frames: bool,
         llm_backend: str, max_minutes: int, api_key: str | None = None,
         display_name: str | None = None) -> None:
    global LATEST_REPORT
    tmpdir = saved_path and os.path.dirname(saved_path)
    try:
        job["status"] = "running"
        with _LOCK:
            report = analyze(source, model, use_frames, llm_backend, max_minutes,
                             progress=lambda m: job["messages"].append(m), api_key=api_key)
        report["file_name"] = display_name or os.path.basename(source)
        try:
            src_p = Path(source).resolve()
            if str(src_p).startswith(str(ROOT_DIR)) and src_p.is_file():
                rel = str(src_p.relative_to(ROOT_DIR)).replace("\\", "/")
                report["media_url"] = f"/api/media?path={rel}"
        except Exception:
            pass
        job["report"] = _serialize(report)
        job["status"] = "done"
        LATEST_REPORT = job["report"]
    except Exception as e:  # noqa: BLE001
        job["error"] = str(e)
        job["status"] = "error"
    finally:
        if tmpdir:
            shutil.rmtree(tmpdir, ignore_errors=True)  # uploaded copies always die with the run


@app.get("/health")
@app.get("/api/health")
def health_check():
    return JSONResponse({"status": "ok", "app": "ClassLens", "version": "1.0.0"})


@app.get("/api/config")
def public_config():
    return {"whisper_model": DEFAULT_WHISPER_MODEL, "llm_backend": DEFAULT_LLM_BACKEND}


@app.get("/")
def index():
    return FileResponse(WEB_PAGE, media_type="text/html")


@app.get("/static/styles.css")
def styles():
    return FileResponse(WEB_DIR / "styles.css", media_type="text/css")


@app.get("/favicon.ico", status_code=204)
def favicon():
    return None


@app.get("/api/samples")
def list_samples():
    """List available local classroom recordings."""
    results = []
    candidates = []
    for base in [ROOT_DIR / "samples", ROOT_DIR / ".tools"]:
        if base.exists():
            for p in base.glob("**/*.*"):
                if p.suffix.lower() in {".mp4", ".wav", ".m4a", ".webm", ".mov"}:
                    candidates.append(p)
    for p in sorted(set(candidates)):
        sz = p.stat().st_size
        results.append({
            "name": p.name,
            "rel_path": str(p.relative_to(ROOT_DIR)).replace("\\", "/"),
            "size_mb": round(sz / (1024 * 1024), 1),
            "ext": p.suffix.lower(),
        })
    return results


@app.get("/api/latest")
def get_latest(_: None = Depends(_require_password)):
    """Return latest analyzed report if available."""
    if LATEST_REPORT:
        return JSONResponse(LATEST_REPORT)
    return JSONResponse({"status": "no_reports_yet"})


@app.get("/api/media")
def stream_media(path: str):
    """Safely stream a local audio or video file from within the workspace."""
    p = (ROOT_DIR / path).resolve()
    if not _is_within_root(p) or not p.is_file():
        raise HTTPException(status_code=404, detail="Media file not found.")
    ext = p.suffix.lower()
    media_types = {
        ".mp4": "video/mp4", ".webm": "video/webm", ".mkv": "video/x-matroska",
        ".wav": "audio/wav", ".mp3": "audio/mpeg", ".m4a": "audio/mp4", ".ogg": "audio/ogg", ".mov": "video/quicktime"
    }
    return FileResponse(p, media_type=media_types.get(ext, "application/octet-stream"))


@app.post("/assess")
async def assess(
    file: UploadFile | None = File(None),
    url: str = Form(""),
    sample: str = Form(""),
    whisper_model: str = Form(DEFAULT_WHISPER_MODEL),
    use_frames: str = Form("true"),
    max_minutes: int = Form(30, ge=1, le=120),
    llm_backend: str = Form(DEFAULT_LLM_BACKEND),
    api_key: str = Form(""),
    token: str = Form(""),
):
    required = os.environ.get("APP_PASSWORD")
    if required and token != required:
        raise HTTPException(status_code=401, detail="Wrong or missing password.")
    url = url.strip()
    sample = sample.strip()
    if not file and not url and not sample:
        raise HTTPException(status_code=400, detail="Provide a file, a link, or pick a sample.")
    if llm_backend not in LLM_BACKENDS:
        raise HTTPException(status_code=400, detail=f"Unknown LLM backend {llm_backend!r}.")

    source, saved = None, None
    display_name = None
    if file:
        tmp = tempfile.mkdtemp(prefix="classlens_upload_")
        display_name = Path(file.filename or "upload.mp4").name
        ext = os.path.splitext(display_name)[1] or ".mp4"
        saved = os.path.join(tmp, "input" + ext)
        with open(saved, "wb") as fh:
            shutil.copyfileobj(file.file, fh)
        file.file.seek(0)
        if os.path.getsize(saved) == 0:
            shutil.rmtree(tmp, ignore_errors=True)
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        source = saved
    elif sample:
        cand = (ROOT_DIR / sample).resolve()
        if not _is_within_root(cand) or not cand.is_file():
            basename = os.path.basename(sample)
            matches = [p for p in ROOT_DIR.glob(f"**/{basename}") if p.is_file() and _is_within_root(p)]
            if matches:
                cand = matches[0]
            else:
                raise HTTPException(status_code=404, detail=f"Sample file {sample!r} not found on server.")
        if cand.suffix.lower() not in {".mp4", ".wav", ".m4a", ".webm", ".mov", ".mkv", ".mp3", ".ogg"}:
            raise HTTPException(status_code=400, detail="Sample must be an audio or video recording.")
        source = str(cand)
    else:
        source = url

    job = {"status": "queued", "messages": ["Assessment queued"], "report": None, "error": None}
    jid = uuid.uuid4().hex[:12]
    JOBS[jid] = job
    threading.Thread(target=_run, args=(job, source, saved, whisper_model, use_frames == "true",
                                        llm_backend, max_minutes, api_key.strip() or None, display_name), daemon=True).start()
    return {"job": jid}


@app.get("/jobs/{job_id}")
def job_status(job_id: str, _: None = Depends(_require_password)):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Unknown job.")
    return JSONResponse(job)


if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host=host, port=port)
